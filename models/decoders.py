from dataclasses import dataclass, asdict

import torch
from torch import nn
from torch.nn import functional as F

from .common import TransformerBlock, LayerNorm

import math

@dataclass
class ViTDecoderConfig():
    latent_dim:         int     = 64

    patch_size:         int     = 4
    num_channels:       int     = 3
    transformer_dim:    int     = 256
    num_heads:          int     = 8
    num_blocks:         int     = 3
    expansion_ratio:    float   = 2.0

    image_size:         int     = 96

    bias:               bool    = False

class ViTDecoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config



        self.num_patches = int((config.image_size*config.image_size) / (config.patch_size*config.patch_size))

        self.projection = nn.Linear(config.latent_dim, config.transformer_dim, bias=config.bias)
        self.pos_emb    = nn.Parameter(0.02 * torch.randn(1, self.num_patches, config.transformer_dim))


        self.blocks     = nn.ModuleList([TransformerBlock(config) for _ in range(config.num_blocks)])
        self.head = nn.Sequential(
            LayerNorm(config),
            nn.Linear(config.transformer_dim, (config.patch_size ** 2) * config.num_channels, bias=config.bias),
            nn.Sigmoid()
        )

    def _patches_to_image(self, patches):
        patches = patches.transpose(1, 2)
        fold = nn.Fold(output_size=(self.config.image_size, self.config.image_size), kernel_size=self.config.patch_size, stride=self.config.patch_size)
        image_reconstructed = fold(patches)
        return image_reconstructed


    def forward(self, latent_representation):
        B, latent_dim = latent_representation.shape

        x = (
            self.projection(latent_representation).unsqueeze(1).repeat(1, self.num_patches, 1) +
            self.pos_emb.repeat(B, 1, 1)
        )


        for block in self.blocks:
            x = block(x)
        return self._patches_to_image(self.head(x))

    def save_checkpoint(self, path):
        torch.save({
            "config": asdict(self.config),
            "state_dict": self.state_dict()
        }, path)

    @classmethod
    def from_checkpoint(cls, path):
        ckpt = torch.load(path)

        model = cls(ViTDecoderConfig(**ckpt["config"]))
        model.load_state_dict(ckpt["state_dict"])

        return model




if __name__ == "__main__":
    pass
