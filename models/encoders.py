import torch
from torch import nn

import numpy as np

from functools import partial

class ImagePreprocessor(nn.Module):
    """
    Converts uint8 (..., 96, 96, 3) images into ImageNet-normalized float32
    (..., 3, 96, 96) tensors. The bottom-band masking is now done by the
    environment wrapper, so this module no longer touches pixels.
    """

    def __init__(self):
        super().__init__()
        self.register_buffer("rgb_mean", torch.tensor([0.485, 0.456, 0.406]))
        self.register_buffer("rgb_std",  torch.tensor([0.229, 0.224, 0.225]))

    def forward(self, images):
        assert isinstance(images, (np.ndarray, torch.Tensor))
        assert images.shape[-3:] == (96, 96, 3)
        assert images.dtype == np.uint8 or images.dtype == torch.uint8

        device = self.rgb_mean.device

        with torch.no_grad():
            if isinstance(images, np.ndarray):
                images = torch.from_numpy(images).to(device, torch.float32) / 255.0
            else:
                images = images.to(device, torch.float32) / 255.0

            images = (images - self.rgb_mean) / self.rgb_std
            images = images.movedim(-1, -3)

        return images



class StateEncoder(nn.Module):
    """
    Encodes an observation (rgb + proprios) into a latent state.

    The forward signature takes a single tensordict-like input with keys
    "rgb" (uint8, shape (..., 96, 96, 3)) and "proprios" (float, shape (..., proprio_dim)).
    Image preprocessing (normalization + channel reordering) is handled internally.
    Arbitrary leading batch dimensions are supported.
    """

    def __init__(self, proprio_dim=7, latent_dim=64):
        super().__init__()

        activation_cls = partial(nn.GELU, approximate="tanh")

        self.image_preprocessor = ImagePreprocessor()

        self.visual_encoder = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=5, padding=2, stride=3),
            activation_cls(),
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=2),
            activation_cls(),
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=2),
            activation_cls(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=2),
            activation_cls(),
            nn.Conv2d(128, 256, kernel_size=4, padding=0, stride=1),
            activation_cls(),
            nn.Flatten(1),
            nn.Linear(256, 256),
        )

        self.proprio_encoder = nn.Sequential(
            nn.Linear(proprio_dim, 256),
            activation_cls(),
            nn.Linear(256, 256),
        )

        self.head = nn.Sequential(
            nn.LayerNorm(512),
            nn.Linear(512, 512),
            activation_cls(),
            nn.Linear(512, latent_dim),
        )

    def forward(self, obs):
        # obs: tensordict / dict with:
        #   "rgb":      (..., 96, 96, 3) uint8
        #   "proprios": (..., proprio_dim) float
        rgb  = obs["rgb"]
        prop = obs["proprios"]

        # Flatten arbitrary leading batch dims so conv/linear see (N, ...).
        batch_dims = rgb.shape[:-3]
        rgb_flat   = rgb.reshape(-1, 96, 96, 3)
        prop_flat  = prop.reshape(-1, prop.shape[-1]).float()

        rgb_flat = self.image_preprocessor(rgb_flat)   # (N, 3, 96, 96) float32

        obs_enc  = self.visual_encoder(rgb_flat)
        prop_enc = self.proprio_encoder(prop_flat)

        x   = torch.cat((obs_enc, prop_enc), -1)
        out = self.head(x)

        return out.reshape(*batch_dims, -1)


if __name__ == "__main__":
    enc = StateEncoder()

    obs = {
        "rgb":      torch.randint(0, 256, (64, 96, 96, 3), dtype=torch.uint8),
        "proprios": torch.randn(64, 7),
    }

    out = enc(obs)
    print(out.shape)

