"""
T2 test fixture: a DI-engine compliant user model (what T3's code generator
produces), implementing the model return contract v2.2:

  compute_actor        -> {'logit': Tensor(B, action_shape)}
  compute_critic       -> {'value': Tensor(B,)}              (squeezed)
  compute_actor_critic -> {'logit': ..., 'value': ...}

The training service exec-loads this file verbatim and requires the class to
register itself as 'user_net'.
"""
import torch
import torch.nn as nn

from ding.utils import MODEL_REGISTRY


@MODEL_REGISTRY.register("user_net")
class UserMLP(nn.Module):
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
            return {"value": self.critic(x).squeeze(-1)}  # (B,) per contract v2.2
        if mode == "compute_actor_critic":
            return {"logit": self.actor(x), "value": self.critic(x).squeeze(-1)}
        raise ValueError(f"unknown mode: {mode}")
