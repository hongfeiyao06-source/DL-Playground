"""
DL-Playground x DI-engine training service (T2, API contract v2.2).

Runner side (dlbackend, py3.11). The backend process cannot `import ding`
(DI-engine lives in a separate py3.10 env), so this module only validates
requests, manages the single-slot training queue and spawns
``training_worker.py`` with ``$DING_PYTHON``. All state exchange with the
worker happens through JSON files under ``backend/trained_models/<task_id>/``:

    config.json   request payload handed to the worker (verbatim model_code)
    status.json   {"status": starting|running|done|error, "train_iter", "eval_reward", "message"}
    curve.json    {"iterations": [...], "eval_rewards": [...], "losses": [...]}
    model.pth     written by the worker on successful completion
    worker.pid    worker process id (for liveness checks)
    worker.log    worker stdout/stderr

Mounted into the main app from ``runner.py``, and also usable standalone:
``uvicorn training_service:app``.
"""
import json
import os
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from fastapi import APIRouter, FastAPI, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel

BACKEND_DIR = Path(__file__).resolve().parent
MODELS_DIR = BACKEND_DIR / "trained_models"
WORKER_SCRIPT = BACKEND_DIR / "training_worker.py"

# The DI-engine python must not be hardcoded (taskcard v2.5): read $DING_PYTHON
# and only fall back to the documented default on this machine.
DEFAULT_DING_PYTHON = r"D:\anaconda3\envs\ding_env\python.exe"

# MVP whitelist (contract v2.2): anything else -> 400.
ALGORITHM_WHITELIST = {"ppo"}
ENV_WHITELIST = {"cartpole"}

MAX_CONCURRENT_TRAININGS = 1  # concurrency queue limit (taskcard v2.5)

RUNNING_STATUSES = ("starting", "running")

# How long POST /start waits for the worker's dummy-forward self-check to
# conclude (torch+ding import takes ~15-30s on CPU). On timeout the task is
# returned anyway and the frontend keeps polling /status.
SELFCHECK_TIMEOUT = 90.0
SELFCHECK_POLL_INTERVAL = 0.5

# Freshly created task dirs without status/pid yet are still occupying the
# queue while the worker interpreter boots.
BOOT_GRACE_SECONDS = 90.0

# task_id -> subprocess.Popen, only valid while this service process lives.
_ACTIVE_PROCS: Dict[str, subprocess.Popen] = {}

# Serializes queue-check + spawn so the limit-1 queue cannot be raced.
_START_LOCK = threading.Lock()


class Hyperparams(BaseModel):
    learning_rate: float = 0.001
    batch_size: int = 256
    max_train_iter: int = 5000  # demo-friendly default (contract v2.2)


class NetworkSpec(BaseModel):
    model_code: str
    obs_shape: Union[int, List[int]]
    action_shape: Union[int, List[int]]


class TrainingStartRequest(BaseModel):
    network: NetworkSpec
    algorithm: str
    env: str
    hyperparams: Optional[Hyperparams] = None


router = APIRouter(prefix="/api/training", tags=["training"])


# ----------------------------------------------------------------- helpers


def _task_dir(task_id: str) -> Path:
    return MODELS_DIR / task_id


def _read_json(path: Path) -> Optional[Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _write_json(path: Path, data: Any) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
    os.replace(str(tmp), str(path))  # same-volume atomic-ish rename


def _pid_alive(pid: int) -> bool:
    """Windows-safe liveness check (os.kill(pid, 0) would TerminateProcess there)."""
    if not pid or pid <= 0:
        return False
    if sys.platform == "win32":
        import ctypes

        STILL_ACTIVE = 259
        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not handle:
            return False
        try:
            exit_code = ctypes.c_ulong()
            if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
                return False
            return exit_code.value == STILL_ACTIVE
        finally:
            kernel32.CloseHandle(handle)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _log_tail(path: Path, chars: int = 2000) -> str:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - chars))
            return f.read()[-chars:]
    except OSError:
        return ""


def _worker_pid(task_id: str) -> Optional[int]:
    pid = _read_json(_task_dir(task_id) / "worker.pid")
    return int(pid) if isinstance(pid, (int, float)) else None


def _require_task(task_id: str) -> Path:
    """Unknown task_id -> 404 for all four read endpoints (contract v2.2)."""
    d = _task_dir(task_id)
    if not d.is_dir():
        raise HTTPException(status_code=404, detail=f"unknown task_id: {task_id}")
    return d


def _task_is_active(d: Path) -> bool:
    """True while a task occupies the single training slot."""
    st = _read_json(d / "status.json")
    if st is None:
        # Worker interpreter still booting: the dir is fresh, count it.
        try:
            return time.time() - d.stat().st_mtime < BOOT_GRACE_SECONDS
        except OSError:
            return False
    if st.get("status") not in RUNNING_STATUSES:
        return False
    pid = _read_json(d / "worker.pid")
    if pid is None:
        status_age = None
        try:
            status_age = time.time() - (d / "status.json").stat().st_mtime
        except OSError:
            pass
        return status_age is not None and status_age < BOOT_GRACE_SECONDS
    return _pid_alive(int(pid))


def _running_task_count() -> int:
    if not MODELS_DIR.is_dir():
        return 0
    return sum(1 for d in MODELS_DIR.iterdir() if d.is_dir() and _task_is_active(d))


def _resolve_status(task_id: str) -> Dict[str, Any]:
    """
    Contract status frame; corrects stale 'starting'/'running' when the
    worker died without writing a final status.json.
    """
    st = _read_json(_task_dir(task_id) / "status.json")
    if st is None:
        # First-frame race (contract v2.2): no status.json yet -> starting.
        return {"status": "starting", "train_iter": 0, "eval_reward": None, "message": ""}
    if st.get("status") in RUNNING_STATUSES:
        proc = _ACTIVE_PROCS.get(task_id)
        alive = proc is not None and proc.poll() is None
        if not alive:
            pid = _worker_pid(task_id)
            alive = pid is not None and _pid_alive(pid)
        if not alive:
            return {
                "status": "error",
                "train_iter": st.get("train_iter", 0),
                "eval_reward": st.get("eval_reward"),
                "message": "training worker exited unexpectedly",
            }
    return st


# ------------------------------------------------------------------ endpoints


@router.post("/start")
def start_training(req: TrainingStartRequest):
    # Lock spans validation + queue check + spawn + self-check wait so the
    # limit-1 queue cannot be raced by concurrent requests.
    with _START_LOCK:
        return _start_training(req)


def _start_training(req: TrainingStartRequest):
    algorithm = (req.algorithm or "").lower()
    env = (req.env or "").lower()
    if algorithm not in ALGORITHM_WHITELIST:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported algorithm '{req.algorithm}'; MVP whitelist: {sorted(ALGORITHM_WHITELIST)}",
        )
    if env not in ENV_WHITELIST:
        raise HTTPException(
            status_code=400,
            detail=f"unsupported env '{req.env}'; MVP whitelist: {sorted(ENV_WHITELIST)}",
        )
    model_code = (req.network.model_code or "").strip()
    if not model_code:
        raise HTTPException(status_code=400, detail="network.model_code is empty")

    def _valid_shape(s: Union[int, List[int]]) -> bool:
        if isinstance(s, bool):
            return False
        if isinstance(s, int):
            return s > 0
        if isinstance(s, (list, tuple)) and len(s) > 0:
            return all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in s)
        return False

    obs_shape = req.network.obs_shape
    action_shape = req.network.action_shape
    if not _valid_shape(obs_shape) or not _valid_shape(action_shape):
        raise HTTPException(
            status_code=400,
            detail="obs_shape/action_shape must be a positive int or a non-empty list of positive ints",
        )
    hp = req.hyperparams or Hyperparams()
    if hp.learning_rate <= 0 or hp.batch_size <= 0 or hp.max_train_iter <= 0:
        raise HTTPException(status_code=400, detail="hyperparams must all be positive")

    if _running_task_count() >= MAX_CONCURRENT_TRAININGS:
        raise HTTPException(
            status_code=409,
            detail="training queue is full (limit 1 concurrent training); stop the running task first",
        )

    task_id = uuid.uuid4().hex[:12]
    task_dir = _task_dir(task_id)
    task_dir.mkdir(parents=True, exist_ok=True)
    config = {
        "task_id": task_id,
        "task_dir": str(task_dir),
        "model_code": model_code,
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "algorithm": req.algorithm,
        "env": req.env,
        "hyperparams": {
            "learning_rate": hp.learning_rate,
            "batch_size": hp.batch_size,
            "max_train_iter": hp.max_train_iter,
        },
    }
    _write_json(task_dir / "config.json", config)

    ding_python = os.environ.get("DING_PYTHON", DEFAULT_DING_PYTHON)
    if not Path(ding_python).is_file():
        raise HTTPException(
            status_code=500,
            detail=f"DING_PYTHON not found: {ding_python} (set the env var to a DI-engine python)",
        )
    log_path = task_dir / "worker.log"
    try:
        log_f = open(log_path, "w", encoding="utf-8", errors="replace")
        proc = subprocess.Popen(
            [ding_python, str(WORKER_SCRIPT), str(task_dir / "config.json")],
            cwd=str(task_dir),
            stdout=log_f,
            stderr=subprocess.STDOUT,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        log_f.close()
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"failed to spawn training worker: {exc}")
    _ACTIVE_PROCS[task_id] = proc

    # Wait for the dummy-forward self-check (contract v2.2): a model that does
    # not satisfy the forward() shape contract must surface as HTTP 400 on
    # /start itself, before any training happens.
    deadline = time.time() + SELFCHECK_TIMEOUT
    while time.time() < deadline:
        st = _read_json(task_dir / "status.json")
        if st is not None:
            if st.get("status") == "running":
                return {"task_id": task_id}
            if st.get("status") == "error":
                raise HTTPException(status_code=400, detail=st.get("message") or "model self-check failed")
        if proc.poll() is not None:
            st = _read_json(task_dir / "status.json") or {}
            if st.get("status") == "error":
                raise HTTPException(status_code=400, detail=st.get("message") or "model self-check failed")
            raise HTTPException(
                status_code=500,
                detail=f"training worker crashed during startup: {_log_tail(log_path)}",
            )
        time.sleep(SELFCHECK_POLL_INTERVAL)
    # Self-check still not concluded (slow imports): hand the task over and let
    # the frontend poll /status, which reports 'starting' in the meantime.
    return {"task_id": task_id}


@router.get("/{task_id}/status")
def get_status(task_id: str):
    _require_task(task_id)
    return _resolve_status(task_id)


@router.post("/{task_id}/stop")
def stop_training(task_id: str):
    _require_task(task_id)
    st = _read_json(_task_dir(task_id) / "status.json") or {}
    if st.get("status") not in RUNNING_STATUSES:
        return {"stopped": True}  # already finished; idempotent
    proc = _ACTIVE_PROCS.get(task_id)
    pid = _worker_pid(task_id)
    alive = (proc is not None and proc.poll() is None) or (pid is not None and _pid_alive(pid))
    if alive:
        if proc is not None:
            try:
                # ding's BaseEnvManager runs envs in-process, so terminating
                # the worker leaves no orphan processes behind.
                proc.terminate()
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
            except OSError:
                pass
        elif sys.platform == "win32":
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(pid)], capture_output=True)
        else:
            try:
                os.kill(int(pid), 9)
            except OSError:
                pass
    _write_json(
        _task_dir(task_id) / "status.json",
        {
            "status": "done",
            "train_iter": st.get("train_iter", 0),
            "eval_reward": st.get("eval_reward"),
            "message": "stopped by user",
        },
    )
    _ACTIVE_PROCS.pop(task_id, None)
    return {"stopped": True}


@router.get("/{task_id}/curve")
def get_curve(task_id: str):
    _require_task(task_id)
    curve = _read_json(_task_dir(task_id) / "curve.json")
    if curve is None:
        # First-frame race (contract v2.2): no curve yet -> empty arrays.
        return {"iterations": [], "eval_rewards": [], "losses": []}
    return curve


@router.get("/{task_id}/model")
def get_model(task_id: str):
    _require_task(task_id)
    model_path = _task_dir(task_id) / "model.pth"
    if not model_path.is_file():
        raise HTTPException(
            status_code=404,
            detail="model not ready: training has not completed successfully (no model.pth yet)",
        )
    return FileResponse(
        str(model_path),
        media_type="application/octet-stream",
        filename=f"cartpole_ppo_{task_id}.pth",
    )


app = FastAPI(title="DL-Playground training service (T2)")
app.include_router(router)
