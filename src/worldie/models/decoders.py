from __future__ import annotations

import torch
from torch import nn


class ObservationDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.project = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, 256 * 4 * 4),
            nn.ReLU(),
        )
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 3, kernel_size=4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, hidden], dim=-1)
        x = self.project(x)
        x = x.view(x.shape[0], 256, 4, 4)
        return self.deconv(x)


class ScalarDecoder(nn.Module):
    def __init__(self, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, latent: torch.Tensor, hidden: torch.Tensor) -> torch.Tensor:
        x = torch.cat([latent, hidden], dim=-1)
        return self.net(x).squeeze(-1)

