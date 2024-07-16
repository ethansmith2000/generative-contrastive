#!/usr/bin/env python
# coding=utf-8
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import copy
import math
import os
import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from accelerate.logging import get_logger
from tqdm.auto import tqdm
import train_utils
from train_utils import (
    init_train_basics,
    log_validation,
    save_model,
    unwrap_model,
    load_models,
    get_optimizer,
    get_dataset,
    more_init,
    resume_model
)
from types import SimpleNamespace
from utils import spherical_add, spherical_noise
import torch.nn.functional as F


default_arguments = dict(
    text_model="openai/clip-vit-base-patch32",
    image_model="openai/clip-vit-base-patch32",
    embed_dim=512,
    z_dim=64,
    w_dim=64,
    df_path="./cc12m.parquet",
    num_validation_images=16,
    output_dir="output",
    seed=123,
    resolution=55,
    train_batch_size=1024,
    max_train_steps=50_000,
    validation_steps=250,
    checkpointing_steps=500,
    checkpoints_total_limit=None,
    resume_from_checkpoint=None,
    gradient_accumulation_steps=1,
    gradient_checkpointing=False,
    # learning_rate=1.0e-4,
    learning_rate=2e-4,
    # learning_rate=2.5e-5,
    lr_scheduler="linear",
    lr_warmup_steps=200,
    lr_num_cycles=1,
    lr_power=1.0,
    dataloader_num_workers=4,
    use_8bit_adam=False,
    adam_beta1=0.9,
    adam_beta2=0.99,
    adam_weight_decay=1e-2,
    adam_epsilon=1e-08,
    max_grad_norm=1.0,
    report_to="wandb",
    mixed_precision="bf16",
    allow_tf32=True,
    logging_dir="logs",
    local_rank=-1,
    num_processes=1,
    use_wandb=True,

    input_noise=0.05,
    observed_noise=0.25,

    mse_loss_scale=0.0,
    tv_loss_scale=0.1,
    contrast_loss_scale=1.0,
    freq_loss_scale=0.1,

    fft_mean_path = "./fft_mean.pt"
)


def tv_loss_fn(inp):
    """L2 total variation loss, as in Mahendran et al."""
    inp = F.pad(inp, (0, 1, 0, 1), 'replicate')
    x_diff = inp[..., :-1, 1:] - inp[..., :-1, :-1]
    y_diff = inp[..., 1:, :-1] - inp[..., :-1, :-1]
    return (x_diff**2 + y_diff**2).mean([1, 2, 3]).mean()


def spectral_noise(inp, mean):
    # what are we doing here
    sample = mean + torch.randn_like(mean) * 0.01 
    sample = torch.exp(sample)
    noise = torch.fft.ifft2(sample).real
    return inp + noise


def freq_loss_fn(inp, mean):
    """L2 loss in the frequency domain."""
    inp = torch.fft.fft2(inp).abs() + 1e-6
    inp = torch.log(inp)
    return F.mse_loss(inp, mean[None,:,:,:])


def train(args):
    logger = get_logger(__name__)
    args = SimpleNamespace(**args)
    accelerator, weight_dtype = init_train_basics(args, logger)

    tokenizer, text_encoder, image_encoder, generator, listener = load_models(args, accelerator, weight_dtype)

    optimizer_generator, lr_scheduler_generator = get_optimizer(args, list(generator.parameters()), accelerator)
    optimizer_listener, lr_scheduler_listener = get_optimizer(args, list(listener.parameters()), accelerator)
    train_dataset, train_dataloader, num_update_steps_per_epoch = get_dataset(args)

    # Prepare everything with our `accelerator`.
    generator, listener, optimizer_generator, optimizer_listener, lr_scheduler_generator, lr_scheduler_listener, train_dataloader = accelerator.prepare(
        generator, listener, optimizer_generator, optimizer_listener, lr_scheduler_generator, lr_scheduler_listener, train_dataloader
    )

    global_step = 0
    if args.resume_from_checkpoint:
        global_step = resume_model(generator, args.resume_from_checkpoint, accelerator)
        global_step = resume_model(listener, args.resume_from_checkpoint, accelerator)

    global_step, first_epoch, progress_bar = more_init(accelerator, args, train_dataloader, train_dataset, logger, num_update_steps_per_epoch, global_step, wandb_name="generative-listener")

    # text_forward = torch.compile(text_encoder.forward)
    text_forward = text_encoder.forward

    # empirically observed power spectra of 10k images at 55x55, in log spaces
    fft_mean = torch.load(args.fft_mean_path).to(text_encoder.device).float()

    for epoch in range(first_epoch, args.num_train_epochs):
        generator.train()
        listener.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(generator):
                texts = batch["text"]
                with torch.no_grad():
                    text_input = tokenizer(texts, return_tensors="pt", padding="max_length", truncation=True, max_length=77).to(text_encoder.device)
                    input_ids = text_input["input_ids"]
                    text_output = text_forward(**text_input).last_hidden_state[torch.arange(input_ids.shape[0]),input_ids.argmax(-1)].float()
                
                # generator
                # scales = torch.ones(text_output.shape[0], device=text_output.device) * args.input_noise
                # noisy_latent = text_output + scales[:,None] * torch.randn_like(text_output)
                zs = torch.randn(text_output.shape[0], args.z_dim, device=text_output.device)
                observed = generator(zs, text_output)

                tv_loss = tv_loss_fn(observed)
                freq_loss = freq_loss_fn(observed.float(), fft_mean)

                # listener
                scales = torch.ones(text_output.shape[0], device=text_output.device) * args.observed_noise
                noisy_observed = observed + scales[:,None,None,None] * torch.randn_like(observed)
                reconstructed = listener(noisy_observed, None)

                normed_text = F.normalize(text_output, p=2, dim=-1).to(reconstructed.dtype)
                normed_recon = F.normalize(reconstructed, p=2, dim=-1).to(reconstructed.dtype)
                sims = (normed_text @ normed_recon.T)
                mse_loss = F.mse_loss(text_output, reconstructed)
                contrast_loss = F.cross_entropy(sims, torch.arange(sims.shape[0]).to(sims.device)) / 2 + F.cross_entropy(sims.T, torch.arange(sims.shape[0]).to(sims.device)) / 2
                acc = (sims.argmax(dim=-1) == torch.arange(sims.shape[0]).to(sims.device)).float().mean()
                loss = contrast_loss * args.contrast_loss_scale + mse_loss * args.mse_loss_scale + tv_loss * args.tv_loss_scale + freq_loss * args.freq_loss_scale
                
                accelerator.backward(loss)
                if accelerator.sync_gradients:
                    generator_norm = accelerator.clip_grad_norm_(generator.parameters(), args.max_grad_norm)
                    listener_norm = accelerator.clip_grad_norm_(listener.parameters(), args.max_grad_norm)
                optimizer_generator.step()
                optimizer_listener.step()
                optimizer_generator.zero_grad(set_to_none=True)
                optimizer_listener.zero_grad(set_to_none=True)
                lr_scheduler_generator.step()
                lr_scheduler_listener.step()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                if accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        save_model(generator, listener,accelerator,save_path, args, logger)
                        

            logs = {"c_loss": contrast_loss.detach().item(), "mse_loss":mse_loss.item(), "acc":acc.item(), 
            "tv_loss": tv_loss.detach().item(), "lr_g": lr_scheduler_generator.get_last_lr()[0], 
            "lr_l": lr_scheduler_listener.get_last_lr()[0], "g_norm": generator_norm.item(), "l_norm": listener_norm.item(),
            "freq_loss": freq_loss.item()}

            progress_bar.set_postfix(**logs)
            if args.use_wandb:
                accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

            if accelerator.is_main_process:
                if global_step % args.validation_steps == 0 and global_step > 0:
                    images = log_validation(
                        generator,
                        listener,
                        tokenizer,
                        text_encoder,
                        texts,
                        weight_dtype,
                        args,   
                        accelerator,
                        epoch=epoch,
                        logger=logger,
                    )

    # Save the lora layers
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, "lora_layers.pth")
        save_model(generator, listener, text_encoder, accelerator, save_path, args, logger)

    accelerator.end_training()


if __name__ == "__main__":
    train(default_arguments)