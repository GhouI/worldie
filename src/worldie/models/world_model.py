from __future__ import annotations

import torch
from torch import nn

from worldie.models.decoders import ObservationDecoder, ScalarDecoder
from worldie.models.encoder import ConvEncoder
from worldie.models.rssm import RSSM


class WorldModel(nn.Module):
    def __init__(self, action_dim: int, latent_dim: int, hidden_dim: int) -> None:
        super().__init__()
        embedding_dim = hidden_dim
        self.encoder = ConvEncoder(embedding_dim=embedding_dim)
        self.rssm = RSSM(
            action_dim=action_dim,
            latent_dim=latent_dim,
            hidden_dim=hidden_dim,
            embedding_dim=embedding_dim,
        )
        self.observation_decoder = ObservationDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.reward_decoder = ScalarDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)
        self.done_decoder = ScalarDecoder(latent_dim=latent_dim, hidden_dim=hidden_dim)

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        observations = batch["observations"]
        actions = batch["actions"]
        batch_size, sequence_plus_one = observations.shape[:2]
        sequence_length = sequence_plus_one - 1
        device = observations.device

        hidden, latent = self.rssm.initial_state(batch_size=batch_size, device=device)
        embeddings = self.encoder(observations[:, 1:].reshape(batch_size * sequence_length, 3, 64, 64))
        embeddings = embeddings.view(batch_size, sequence_length, -1)

        reconstructions: list[torch.Tensor] = []
        rewards: list[torch.Tensor] = []
        dones: list[torch.Tensor] = []
        prior_kls: list[torch.Tensor] = []

        for step in range(sequence_length):
            hidden, latent, prior_dist, posterior_dist = self.rssm.observe_step(
                prev_hidden=hidden,
                prev_latent=latent,
                action=actions[:, step],
                embedding=embeddings[:, step],
            )
            reconstructions.append(self.observation_decoder(latent, hidden))
            rewards.append(self.reward_decoder(latent, hidden))
            dones.append(self.done_decoder(latent, hidden))
            prior_kls.append(torch.distributions.kl_divergence(posterior_dist, prior_dist))

        return {
            "reconstruction": torch.stack(reconstructions, dim=1),
            "reward": torch.stack(rewards, dim=1),
            "done_logits": torch.stack(dones, dim=1),
            "kl": torch.stack(prior_kls, dim=1),
        }

