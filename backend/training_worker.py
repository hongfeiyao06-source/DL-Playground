"""
DI-engine training worker (T2, API contract v2.2). Runs under $DING_PYTHON.

Usage: training_worker.py <config.json>

The runner (dlbackend, py3.11) cannot `import ding`, so it spawns this script
as a subprocess. This worker:

  1. exec-loads the user's ``model_code`` verbatim (never modified) and expects
     it to register its model class under ``MODEL_REGISTRY`` name 'user_net';
  2. runs a dummy-forward self-check in all three modes defined by the model
     return contract v2.2 (compute_actor / compute_critic /
     compute_actor_critic) and aborts immediately on shape violations;
  3. trains PPO on CartPole with an explicit ``model=dict(type='user_net')``
     config, Windows-compatible env manager ('base', collector_env_num=4);
  4. records curve data: losses through the official learner hook mechanism
     (a custom after_iter LearnerHook reading the learner's log_buffer), eval
     rewards through InteractionSerialEvaluator.eval()'s return value — the
     evaluator exposes no hook extension points in ding v0.5.3 (verified in
     ding/worker/collector/interaction_serial_evaluator.py);
  5. saves model.pth on successful completion.

State files (in the task dir): status.json, curve.json, model.pth, worker.pid.
Exit codes: 0=done, 2=self-check failed, 3=training error.
"""
import json
import os
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------- state files

_task_dir: Path = None
_curve = {"iterations": [], "eval_rewards": [], "losses": []}
_loss_buf = []  # per-train-iteration total_loss, pushed by CurveLogHook
_last_status_write = 0.0
_last_eval_reward = None
_last_train_iter = 0


def _atomic_write_json(path: Path, data) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(str(tmp), str(path))


def write_status(status: str, train_iter: int = 0, eval_reward=None, message: str = "") -> None:
    global _last_status_write
    _last_status_write = time.time()
    _atomic_write_json(
        _task_dir / "status.json",
        {"status": status, "train_iter": int(train_iter), "eval_reward": eval_reward, "message": message},
    )


def write_curve() -> None:
    _atomic_write_json(_task_dir / "curve.json", _curve)


def record_eval(train_iter: int, eval_reward: float) -> None:
    """Append one curve row: the loss is the mean total_loss since the last row
    (losses are collected per-iteration by the CurveLogHook)."""
    global _last_eval_reward
    loss = float(sum(_loss_buf) / len(_loss_buf)) if _loss_buf else 0.0
    _loss_buf.clear()
    _last_eval_reward = round(float(eval_reward), 4)
    _curve["iterations"].append(int(train_iter))
    _curve["eval_rewards"].append(_last_eval_reward)
    _curve["losses"].append(round(loss, 4))
    write_curve()
    write_status("running", int(train_iter), _last_eval_reward)


# ------------------------------------------------------------------ self-check


def _fmt_output(out) -> str:
    if isinstance(out, dict):
        return "{" + ", ".join(
            f"{k}: Tensor{tuple(v.shape)}" if hasattr(v, "shape") else f"{k}: {type(v).__name__}"
            for k, v in out.items()
        ) + "}"
    return f"{type(out).__name__}"


def dummy_forward_selfcheck(obs_shape, action_shape) -> None:
    """
    Contract v2.2 self-check, run BEFORE training:
      compute_actor        -> {'logit': Tensor(B, action_shape)}
      compute_critic       -> {'value': Tensor(B,)}   (squeezed, NOT (B, 1))
      compute_actor_critic -> {'logit': ..., 'value': ...}
    Any violation aborts training and surfaces as HTTP 400 on /start.
    """
    import torch
    from ding.utils import MODEL_REGISTRY

    if "user_net" not in MODEL_REGISTRY:
        raise RuntimeError(
            "model_code did not register any class under MODEL_REGISTRY name 'user_net' "
            "(expected: @MODEL_REGISTRY.register('user_net'))"
        )
    model_cls = MODEL_REGISTRY["user_net"]
    try:
        model = model_cls(obs_shape=obs_shape, action_shape=action_shape, action_space="discrete")
    except Exception as exc:
        raise RuntimeError(f"failed to instantiate user_net(obs_shape={obs_shape!r}, "
                           f"action_shape={action_shape!r}, action_space='discrete'): {exc}")
    model.eval()

    obs = torch.randn(4, obs_shape) if isinstance(obs_shape, int) else torch.randn(4, *obs_shape)
    expected_logit = (4, action_shape) if isinstance(action_shape, int) else (4, *action_shape)

    with torch.no_grad():
        out = model(obs, mode="compute_actor")
        if not isinstance(out, dict) or "logit" not in out or not hasattr(out["logit"], "shape"):
            raise RuntimeError(
                f"compute_actor must return {{'logit': Tensor(B, action_shape)}}; got {_fmt_output(out)}"
            )
        if tuple(out["logit"].shape) != expected_logit:
            raise RuntimeError(
                f"compute_actor logit shape {tuple(out['logit'].shape)} != expected {expected_logit} "
                f"(B={4}, action_shape={action_shape})"
            )

        out = model(obs, mode="compute_critic")
        if not isinstance(out, dict) or "value" not in out or not hasattr(out["value"], "shape"):
            raise RuntimeError(
                f"compute_critic must return {{'value': Tensor(B,)}}; got {_fmt_output(out)}"
            )
        if tuple(out["value"].shape) != (4,):
            raise RuntimeError(
                f"compute_critic value shape {tuple(out['value'].shape)} != expected (4,) "
                f"— the critic head must be squeezed to 1-D, not (B, 1)"
            )

        out = model(obs, mode="compute_actor_critic")
        if not isinstance(out, dict) or "logit" not in out or "value" not in out:
            raise RuntimeError(
                f"compute_actor_critic must return {{'logit': ..., 'value': ...}}; got {_fmt_output(out)}"
            )
        if tuple(out["logit"].shape) != expected_logit or tuple(out["value"].shape) != (4,):
            raise RuntimeError(
                f"compute_actor_critic shapes wrong: logit {tuple(out['logit'].shape)} "
                f"(expected {expected_logit}), value {tuple(out['value'].shape)} (expected (4,))"
            )


# ------------------------------------------------------- learner hook (curve)

def _install_curve_hook() -> None:
    """Register a custom after_iter learner hook (official mechanism from
    ding/worker/learner/learner_hook.py): it snapshots the per-iteration
    total loss from the learner's log_buffer before LogShowHook clears it
    (priority 10 < log_show's 20). The config wires it in via
    policy.learn.learner.hook.curve_log."""
    from ding.worker.learner.learner_hook import LearnerHook, register_learner_hook

    class _CurveLogHook(LearnerHook):
        def __call__(self, engine) -> None:
            if engine.rank != 0:
                return
            scalars = engine.log_buffer.get("scalar", {})
            loss = scalars.get("total_loss")
            if loss is None:
                return
            try:
                _loss_buf.append(float(loss))
            except (TypeError, ValueError):
                pass

    register_learner_hook("curve_log", _CurveLogHook)


def build_configs(cfg_input: dict):
    """Assemble main/create configs, copying dizoo's cartpole_ppo_config
    structure with an explicit model type (contract v2.2)."""
    from easydict import EasyDict

    obs_shape = cfg_input["obs_shape"]
    action_shape = cfg_input["action_shape"]
    hp = cfg_input["hyperparams"]

    main_config = EasyDict(
        dict(
            exp_name="exp",
            env=dict(
                collector_env_num=4,  # Windows-friendly (taskcard v2.5)
                evaluator_env_num=4,
                n_evaluator_episode=5,
                stop_value=195,  # contract v2.2: early stop on eval_reward >= 195
            ),
            policy=dict(
                cuda=False,
                action_space="discrete",
                # Explicit type: never rely on PPO's internal 'vac' default.
                model=dict(type="user_net", obs_shape=obs_shape, action_shape=action_shape, action_space="discrete"),
                learn=dict(
                    epoch_per_collect=2,
                    batch_size=int(hp["batch_size"]),
                    learning_rate=float(hp["learning_rate"]),
                    value_weight=0.5,
                    entropy_weight=0.01,
                    clip_ratio=0.2,
                    # custom after_iter hook -> curve losses (official hook
                    # mechanism, see learner_hook.py build_learner_hook_by_cfg)
                    learner=dict(
                        hook=dict(
                            curve_log=dict(
                                type="curve_log",
                                name="curve_log",
                                position="after_iter",
                                priority=10,
                            )
                        )
                    ),
                ),
                collect=dict(n_sample=256, unroll_len=1, discount_factor=0.9, gae_lambda=0.95),
                eval=dict(evaluator=dict(eval_freq=100)),
            ),
        )
    )
    create_config = EasyDict(
        dict(
            env=dict(type="cartpole", import_names=["dizoo.classic_control.cartpole.envs.cartpole_env"]),
            env_manager=dict(type="base"),
            policy=dict(type="ppo"),
        )
    )
    # Windows compatibility (taskcard v2.5): base env manager.
    create_config.env.manager = EasyDict(type="base")
    return main_config, create_config


def run_training(main_config, create_config, max_train_iter: int):
    """
    The serial on-policy training loop from
    ding/entry/serial_entry_onpolicy.py, instrumented with curve/status
    reporting. Eval rewards come from evaluator.eval()'s return value (the
    evaluator has no hook extension points in ding v0.5.3); losses come from
    the CurveLogHook. Component APIs are used exactly as the entry does.
    """
    import numpy as np
    from functools import partial
    from tensorboardX import SummaryWriter

    from ding.config import compile_config
    from ding.envs import get_vec_env_setting, create_env_manager
    from ding.policy import create_policy
    from ding.utils import set_pkg_seed, get_rank
    from ding.worker import (
        BaseLearner,
        InteractionSerialEvaluator,
        BaseSerialCommander,
        create_serial_collector,
    )

    create_config.policy.type = create_config.policy.type + "_command"
    cfg = compile_config(
        main_config,
        seed=0,
        env=None,
        auto=True,
        create_cfg=create_config,
        save_cfg=True,
        renew_dir=True,
    )
    env_fn, collector_env_cfg, evaluator_env_cfg = get_vec_env_setting(cfg.env)
    collector_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in collector_env_cfg])
    evaluator_env = create_env_manager(cfg.env.manager, [partial(env_fn, cfg=c) for c in evaluator_env_cfg])
    collector_env.seed(cfg.seed)
    evaluator_env.seed(cfg.seed, dynamic_seed=False)
    set_pkg_seed(cfg.seed, use_cuda=cfg.policy.cuda)
    policy = create_policy(cfg.policy, model=None, enable_field=["learn", "collect", "eval", "command"])

    tb_logger = SummaryWriter(os.path.join("./{}/log/".format(cfg.exp_name), "serial")) if get_rank() == 0 else None
    learner = BaseLearner(cfg.policy.learn.learner, policy.learn_mode, tb_logger, exp_name=cfg.exp_name)
    collector = create_serial_collector(
        cfg.policy.collect.collector, env=collector_env, policy=policy.collect_mode, tb_logger=tb_logger, exp_name=cfg.exp_name
    )
    evaluator = InteractionSerialEvaluator(
        cfg.policy.eval.evaluator, evaluator_env, policy.eval_mode, tb_logger, exp_name=cfg.exp_name
    )
    commander = BaseSerialCommander(
        cfg.policy.other.commander, learner, collector, evaluator, None, policy.command_mode
    )

    max_env_step = int(1e10)  # same default as serial_pipeline_onpolicy
    stop = False
    learner.call_hook("before_run")
    try:
        while True:
            collect_kwargs = commander.step()
            if evaluator.should_eval(learner.train_iter):
                stop, eval_info = evaluator.eval(learner.save_checkpoint, learner.train_iter, collector.envstep)
                if eval_info and "eval_episode_return" in eval_info:
                    record_eval(learner.train_iter, float(np.mean(eval_info["eval_episode_return"])))
                if stop:
                    break
            new_data = collector.collect(train_iter=learner.train_iter, policy_kwargs=collect_kwargs)
            learner.train(new_data, collector.envstep)
            global _last_train_iter, _last_status_write
            _last_train_iter = learner.train_iter
            if time.time() - _last_status_write >= 2.0:
                write_status("running", learner.train_iter, _last_eval_reward)
            if collector.envstep >= max_env_step or learner.train_iter >= max_train_iter:
                break
    finally:
        learner.call_hook("after_run")
    return policy, stop


def load_model_code(code: str) -> None:
    """exec the user's model_code verbatim (contract v2.2: load only, never modify)."""
    import torch
    import torch.nn as nn
    from ding.utils import MODEL_REGISTRY

    ns = {"__builtins__": __builtins__, "torch": torch, "nn": nn, "MODEL_REGISTRY": MODEL_REGISTRY}
    try:
        exec(code, ns)  # noqa: S102 - user code is executed by design
    except Exception as exc:
        raise RuntimeError(f"model_code failed to execute: {exc}")


def main() -> int:
    global _task_dir
    if len(sys.argv) != 2:
        print("Usage: python training_worker.py <config.json>")
        return 1

    config_path = Path(sys.argv[1])
    with open(config_path, "r", encoding="utf-8") as f:
        cfg_input = json.load(f)

    _task_dir = Path(cfg_input["task_dir"])
    os.chdir(str(_task_dir))
    with open(_task_dir / "worker.pid", "w") as f:
        f.write(str(os.getpid()))
    write_status("starting", 0, None, "worker started")

    # ---- phase 1: load + self-check (failures must surface as HTTP 400) ----
    try:
        load_model_code(cfg_input["model_code"])
        dummy_forward_selfcheck(cfg_input["obs_shape"], cfg_input["action_shape"])
    except Exception as exc:
        write_status("error", 0, None, f"model self-check failed: {exc}")
        return 2

    # ---- phase 2: training ----
    write_status("running", 0, None, "self-check passed; preparing training environment")
    try:
        _install_curve_hook()
        main_config, create_config = build_configs(cfg_input)
        policy, stopped_early = run_training(main_config, create_config, int(cfg_input["hyperparams"]["max_train_iter"]))

        state_dict = policy.learn_mode.state_dict()
        import torch

        torch.save(state_dict, str(_task_dir / "model.pth"))

        message = "training finished"
        if stopped_early:
            message += " (eval_reward reached stop_value)"
        write_status("done", _last_train_iter, _last_eval_reward, message)
        return 0
    except Exception:
        import traceback

        traceback.print_exc()
        write_status("error", _last_train_iter, _last_eval_reward, f"training failed: see worker.log")
        return 3


if __name__ == "__main__":
    sys.exit(main())
