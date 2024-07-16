import torch
from torch import nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, z_dim, im_dim):
        super().__init__()
        dim_1 = int(z_dim * 1.5)
        dim_2 = int(z_dim)
        dim_3 = int(z_dim // 2)
        dim_4 = int(z_dim // 3)
        dim_5 = int(z_dim // 4)
        self.gen = [
            self.make_gen_block(z_dim, dim_1),
            self.make_gen_block(dim_1, dim_2, kernel_size=4, stride=1),
            self.make_gen_block(dim_2, dim_3),
            self.make_gen_block(dim_3, dim_4),
            self.make_gen_block(dim_4, im_dim, final_layer=True),
            # self.make_gen_block(dim_5, im_dim, final_layer=True),
        ]
        self.gen = nn.ModuleList(self.gen)

    def make_gen_block(self, input_channels, output_channels, kernel_size=3, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
                nn.GroupNorm(32, output_channels),
                nn.ReLU(),
            )
        else:
            return nn.Sequential(
                nn.ConvTranspose2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, noise):
        noise = noise.view(len(noise), -1, 1, 1)
        for i in range(len(self.gen)):
            noise = self.gen[i](noise)
        return noise



class Listener(nn.Module):

    def __init__(self, im_dim, z_dim):
        super().__init__()
        dim_1 = int(z_dim // 4)
        dim_2 = int(z_dim // 3)
        dim_3 = int(z_dim // 2)
        dim_4 = int(z_dim)
        dim_5 = int(z_dim * 1.5)
        self.crit = [
            self.make_critic_block(im_dim, dim_1),
            self.make_critic_block(dim_1, dim_2),
            self.make_critic_block(dim_2, dim_3),
            # self.make_critic_block(dim_4, dim_5),
            self.make_critic_block(dim_3, z_dim, final_layer=True),
        ]
        self.crit = nn.ModuleList(self.crit)

    def make_critic_block(self, input_channels, output_channels, kernel_size=4, stride=2, final_layer=False):
        if not final_layer:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
                nn.GroupNorm(32, output_channels),
                nn.LeakyReLU(0.2),
            )
        else:
            return nn.Sequential(
                nn.Conv2d(input_channels, output_channels, kernel_size, stride),
            )

    def forward(self, image):
        for i in range(len(self.crit)):
            image = self.crit[i](image)
        crit_pred = image
        return crit_pred.view(len(crit_pred), -1)