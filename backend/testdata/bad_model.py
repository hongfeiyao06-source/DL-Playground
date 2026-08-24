"""
T2 test fixture: deliberately violates the model return contract v2.2 — the
critic head is NOT squeezed, so compute_critic returns (B, 1) instead of (B,).
POST /api/training/start must answer 400 after the dummy-forward self-check.
"""
import torch
import torch.nn as nn

from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register("user_net")
class BadCriticMLP(nn.Module):
    mode = ["compute_actor", "compute_critic", "compute_actor_critic"]

    def __init__(self, obs_shape=4, action_shape=2, action_space="discrete"):
        super().__init__()
        self.actor = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, action_shape),
        )
        self.critic = nn.Sequential(
            nn.Linear(obs_shape, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, x, mode="compute_actor"):
        if mode == "compute_actor":
            return {"logit": self.actor(x)}
        if mode == "compute_critic":
            return {"value": self.critic(x)}  # (B, 1) -> self-check must fail
        if mode == "compute_actor_critic":
            return {"logit": self.actor(x), "value": self.critic(x)}
        raise ValueError(f"unknown mode: {mode}")
