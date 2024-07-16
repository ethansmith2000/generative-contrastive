
from pathlib import Path
import os
import numpy as np
import torch
import torch.utils.checkpoint
import transformers
from accelerate import Accelerator
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from accelerate.utils import ProjectConfiguration, set_seed
import diffusers
from diffusers.utils.torch_utils import is_compiled_module
import wandb
import logging
import math
# import common.loras as loras
# from loras import patch_lora
import sys
sys.path.append('..')
import random
from diffusers import DPMSolverMultistepScheduler
from diffusers.optimization import get_scheduler
from transformers import AutoTokenizer, PretrainedConfig

import diffusers
from diffusers import (
    AutoencoderKL,
    DDPMScheduler,
    DiffusionPipeline,
    DPMSolverMultistepScheduler,
    StableDiffusionPipeline,
    UNet2DConditionModel,
)
import copy
from tqdm import tqdm

import numpy as np

# from models import Generator, Listener
from networks_stylegan2 import Listener
from networks_stylegan3 import Generator
import pandas as pd


class TextDataset(Dataset):

    def __init__(self, texts):
        self.texts = texts

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        example = {"text": self.texts[index]}
        return example



def save_model(generator, listener, accelerator, save_path, args, logger):
    full_state_dict ={
        "generator": generator.state_dict(),
        "listener": listener.state_dict(),
    }

    torch.save(full_state_dict, save_path)
    logger.info(f"Saved state to {save_path}")


def unwrap_model(accelerator, model):
    model = accelerator.unwrap_model(model)
    model = model._orig_mod if is_compiled_module(model) else model
    return model


def log_validation(
    generator,
    listener,
    tokenizer,
    text_encoder,
    texts,
    weight_dtype,
    args,
    accelerator,
    epoch,
    logger,
    is_final_validation=False,
):

    logger.info(
        f"Running validation... \n Generating {args.num_validation_images} images:"
    )

    texts = texts[:args.num_validation_images]

    with torch.cuda.amp.autocast():
        toks = tokenizer(texts, return_tensors="pt", padding="max_length", max_length=77, truncation=True).to(text_encoder.device)
        text_output = text_encoder(**toks).pooler_output

        noisy_latent = text_output + 0.1 * torch.randn_like(text_output)
        observed = generator(noisy_latent)

        observed = observed.detach().cpu().permute(0,2,3,1).numpy() * 0.5 + 0.5
        observed = np.clip(observed, 0, 1) * 255
        observed = observed.astype(np.uint8)
        print(observed.shape)
        images = [Image.fromarray(observed[i]) for i in range(observed.shape[0])]


    for tracker in accelerator.trackers:
        if args.use_wandb:
            phase_name = "test" if is_final_validation else "validation"
            if tracker.name == "tensorboard":
                np_images = np.stack([np.asarray(img) for img in images])
                tracker.writer.add_images(phase_name, np_images, epoch, dataformats="NHWC")
            if tracker.name == "wandb":
                tracker.log(
                    {
                        phase_name: [
                            wandb.Image(image, caption=f"{i}") for i, image in enumerate(images)
                        ]
                    }
                )

    torch.cuda.empty_cache()

    return images


def init_train_basics(args, logger):
    logging_dir = Path(args.output_dir, args.logging_dir)
    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=logging_dir)
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Enable TF32 for faster training on Ampere GPUs,
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    return accelerator, weight_dtype


def load_models(args, accelerator, weight_dtype):
    # Load the text_encoder
    tokenizer = AutoTokenizer.from_pretrained(args.text_model)
    text_encoder = transformers.CLIPTextModelWithProjection.from_pretrained(args.text_model).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)
    image_encoder = transformers.CLIPVisionModelWithProjection.from_pretrained(args.image_model).requires_grad_(False).to(accelerator.device, dtype=weight_dtype)

    # generator = Generator(z_dim=args.dim, im_dim=3).to(accelerator.device, dtype=weight_dtype)
    # listener = Listener(im_dim=3, z_dim=args.dim).to(accelerator.device, dtype=weight_dtype)

    generator = Generator(z_dim=args.z_dim, c_dim=args.embed_dim, w_dim=args.w_dim, img_resolution=args.resolution, img_channels=3,).to(accelerator.device, dtype=weight_dtype)
    listener = Listener(c_dim=args.embed_dim, img_resolution=args.resolution, img_channels=3).to(accelerator.device, dtype=weight_dtype)

    # if args.gradient_checkpointing:
    #     unet.enable_gradient_checkpointing()

    return tokenizer, text_encoder, image_encoder, generator, listener


def get_optimizer(args, params_to_optimize, accelerator):
    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    optimizer_class = torch.optim.AdamW
    if args.use_8bit_adam:
        import bitsandbytes as bnb
        optimizer_class = bnb.optim.AdamW8bit

    optimizer = optimizer_class(
        params_to_optimize,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=args.max_train_steps * accelerator.num_processes,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )

    return optimizer, lr_scheduler


def get_dataset(args):
    df = pd.read_parquet(args.df_path)
    texts = df['text'].tolist()

    # Dataset and DataLoaders creation:
    train_dataset = TextDataset(texts)

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=True,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    overrode_max_train_steps = False
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)

    return train_dataset, train_dataloader, num_update_steps_per_epoch


def resume_model(model, path, accelerator):
    accelerator.print(f"Resuming from checkpoint {path}")
    global_step = int(path.split("-")[-1])
    state_dict = torch.load(path, map_location="cpu")

    if isinstance(model, Generator):
        model.load_state_dict(state_dict["generator"])
    elif isinstance(model, Listener):
        model.load_state_dict(state_dict["listener"])

    return global_step


def more_init(accelerator, args, train_dataloader, train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="diffusion_lora"):
    if accelerator.is_main_process:
        tracker_config = vars(copy.deepcopy(args))
        if args.use_wandb:
            accelerator.init_trackers(wandb_name, config=tracker_config)

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")

    initial_global_step = global_step
    first_epoch = global_step // num_update_steps_per_epoch

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        disable=not accelerator.is_local_main_process,
    )

    return global_step, first_epoch, progress_bar