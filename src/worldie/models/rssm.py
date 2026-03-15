from __future__ import annotations

import torch
from torch import nn
from torch.distributions import Independent, Normal


def _normal_from_stats(mean: torch.Tensor, std_param: torch.Tensor) -> Independent:
    std = torch.nn.functional.softplus(std_param) + 0.1
    return Independent(Normal(mean, std), 1)


class RSSM(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.action_embedding = nn.Embedding(action_dim, hidden_dim)
        self.gru = nn.GRUCell(latent_dim + hidden_dim, hidden_dim)
        self.prior = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.posterior = nn.Sequential(
            nn.Linear(hidden_dim + embedding_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2 * latent_dim),
        )
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim

    def initial_state(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        latent = torch.zeros(batch_size, self.latent_dim, device=device)
        return hidden, latent

    def observe_step(
        self,
        prev_hidden: torch.Tensor,
        prev_latent: torch.Tensor,
        action: torch.Tensor,
        embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, Independent, Independent]:
        action_embed = self.action_embedding(action)
        hidden_input = torch.cat([prev_latent, action_embed], dim=-1)
        hidden = self.gru(hidden_input, prev_hidden)

        prior_stats = self.prior(hidden)
        prior_mean, prior_std_param = torch.chunk(prior_stats, 2, dim=-1)
        prior_dist = _normal_from_stats(prior_mean, prior_std_param)

        posterior_input = torch.cat([hidden, embedding], dim=-1)
        posterior_stats = self.posterior(posterior_input)
        post_mean, post_std_param = torch.chunk(posterior_stats, 2, dim=-1)
        posterior_dist = _normal_from_stats(post_mean, post_std_param)
        latent = posterior_dist.rsample()

        return hidden, latent, prior_dist, posterior_dist

