"""Shape assertions for the DI-engine model emitted by generateDiEngineModel.

This reads the generator's output fixture (NOT hand-written sample code), execs
it, instantiates the model, and verifies the discrete actor-critic return
structure against the contract:

    compute_actor        -> {'logit': (B, action_shape)}   (raw logits, no tanh)
    compute_critic       -> {'value': (B,)}                (squeezed to 1-D)
    compute_actor_critic -> {'logit': ..., 'value': ...}

Run after the node test has produced the fixtures:

    npm test                       # writes fixtures/sample_model.py
    $DING_PYTHON diengine_model_test.py
"""

import json
import os

import torch
from ding.utils import MODEL_REGISTRY

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURES = os.path.join(HERE, "fixtures")


def main() -> None:
    model_path = os.path.join(FIXTURES, "sample_model.py")
    shapes_path = os.path.join(FIXTURES, "sample_shapes.json")

    with open(model_path, "r", encoding="utf-8") as f:
        model_code = f.read()

    namespace: dict = {}
    exec(compile(model_code, model_path, "exec"), namespace)

    with open(shapes_path, "r", encoding="utf-8") as f:
        shapes = json.load(f)

    obs_shape = int(shapes["obs_shape"])
    action_shape = int(shapes["action_shape"])

    model = MODEL_REGISTRY["user_net"](
        obs_shape=obs_shape, action_shape=action_shape, action_space="discrete"
    )
    batch = 7
    x = torch.randn(batch, obs_shape)

    actor = model(x, "compute_actor")
    assert set(actor.keys()) == {"logit"}, actor.keys()
    assert actor["logit"].shape == (batch, action_shape), actor["logit"].shape

    critic = model(x, "compute_critic")
    assert set(critic.keys()) == {"value"}, critic.keys()
    assert critic["value"].shape == (batch,), critic["value"].shape

    actor_critic = model(x, "compute_actor_critic")
    assert set(actor_critic.keys()) == {"logit", "value"}, actor_critic.keys()
    assert actor_critic["logit"].shape == (batch, action_shape), actor_critic["logit"].shape
    assert actor_critic["value"].shape == (batch,), actor_critic["value"].shape

    print("diengine_model_test: OK")
    print(
        "obs_shape={obs} action_shape={act} "
        "logit={logit} value={value}".format(
            obs=obs_shape,
            act=action_shape,
            logit=tuple(actor["logit"].shape),
            value=tuple(critic["value"].shape),
        )
    )


if __name__ == "__main__":
    main()
