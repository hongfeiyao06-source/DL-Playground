#!/usr/bin/env bash
# =============================================================================
# T2 acceptance test: full training-service chain (API contract v2.2).
#
#   start -> status -> curve -> stop -> model
#
# Covers: whitelist 400, dummy-forward self-check 400, queue limit 409,
# unknown task_id 404, stop semantics, real curve data (learner hook),
# model.pth download.
#
# Usage:
#   uvicorn training_service:app --host 0.0.0.0 --port 8000   (dlbackend env)
#   ./test_training.sh
#
# Env overrides: TRAINING_BASE (default http://localhost:8000),
#                PYTHON (any python3 for JSON helpers, default: dlbackend env).
# =============================================================================
set -u
cd "$(dirname "$0")"

BASE="${TRAINING_BASE:-http://localhost:8000}"
PY="${PYTHON:-python}"
command -v "$PY" >/dev/null 2>&1 || PY="/d/anaconda3/envs/dlbackend/python.exe"

TMP="$(mktemp -d)"
RESP="$TMP/resp.json"
BODY="$TMP/body.json"
MODELDL="$TMP/downloaded.pth"
PASS=0
FAIL=0

say() { printf '%s\n' "$*"; }
ok()  { PASS=$((PASS + 1)); say "  [PASS] $*"; }
bad() { FAIL=$((FAIL + 1)); say "  [FAIL] $*"; }

# req <method> <path> [bodyfile] -> writes $RESP, echoes HTTP status code
req() {
  local method="$1" path="$2" body="${3:-}"
  if [ -n "$body" ]; then
    curl -s -o "$RESP" -w "%{http_code}" --noproxy "localhost,127.0.0.1" -X "$method" "$BASE$path" \
      -H "Content-Type: application/json" --data-binary "@$body"
  else
    curl -s -o "$RESP" -w "%{http_code}" --noproxy "localhost,127.0.0.1" -X "$method" "$BASE$path"
  fi
}

# jget <key> -> value from $RESP (lowercased for booleans/status)
jget() {
  "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(str(d.get(sys.argv[2],'')).lower())" "$RESP" "$1"
}

# jget_raw <key> -> value from $RESP, original case (for human-readable details)
jget_raw() {
  "$PY" -c "import json,sys;d=json.load(open(sys.argv[1],encoding='utf-8'));print(d.get(sys.argv[2],''))" "$RESP" "$1"
}

# make_body <modelfile> <algorithm> <env> <max_train_iter> -> $BODY
make_body() {
  "$PY" - "$1" "$2" "$3" "$4" > "$BODY" <<'PYEOF'
import json, sys
model_file, algo, env, iters = sys.argv[1:5]
code = open(model_file, encoding="utf-8").read()
print(json.dumps({
    "network": {"model_code": code, "obs_shape": 4, "action_shape": 2},
    "algorithm": algo,
    "env": env,
    "hyperparams": {"learning_rate": 0.001, "batch_size": 256, "max_train_iter": int(iters)},
}))
PYEOF
}

expect() { # <desc> <want> <got>
  if [ "$3" = "$2" ]; then ok "$1 ($3)"; else bad "$1: wanted $2 got $3"; fi
}

# poll_status <task_id> <want> <timeout_secs>; 0 on match, 1 on error/timeout
poll_status() {
  local tid="$1" want="$2" timeout="${3:-900}" t0 st code
  t0=$(date +%s)
  while :; do
    code=$(req GET "/api/training/$tid/status")
    st=$(jget status)
    [ "$st" = "$want" ] && return 0
    if [ "$st" = "error" ]; then
      say "     status=error detail: $(jget message)"
      return 1
    fi
    [ $(( $(date +%s) - t0 )) -ge "$timeout" ] && return 1
    sleep 2
  done
}

say "== 1. server up + unknown task_id -> 404 on all four endpoints =="
expect "GET  /status unknown" 404 "$(req GET /api/training/no_such_task/status)"
expect "POST /stop   unknown" 404 "$(req POST /api/training/no_such_task/stop)"
expect "GET  /curve  unknown" 404 "$(req GET /api/training/no_such_task/curve)"
expect "GET  /model  unknown" 404 "$(req GET /api/training/no_such_task/model)"

say "== 2. MVP whitelist: algorithm/env outside [ppo, cartpole] -> 400 =="
make_body testdata/sample_model.py sac cartpole 100
expect "algorithm=sac    -> 400" 400 "$(req POST /api/training/start "$BODY")"
make_body testdata/sample_model.py ppo pendulum 100
expect "env=pendulum     -> 400" 400 "$(req POST /api/training/start "$BODY")"

say "== 3. dummy-forward self-check: critic returns (B,1) -> 400 on /start =="
make_body testdata/bad_model.py ppo cartpole 100
expect "bad critic shape -> 400" 400 "$(req POST /api/training/start "$BODY")"
say "     detail: $(jget_raw detail)"

say "== 4. real training: ppo + cartpole, max_train_iter=3000 =="
make_body testdata/sample_model.py ppo cartpole 3000
expect "start            -> 200" 200 "$(req POST /api/training/start "$BODY")"
TID=$(jget task_id)
say "     task_id=$TID"
if poll_status "$TID" done 900; then
  ok "status reached 'done'"
else
  bad "status never reached 'done'"
fi
_=$(req GET "/api/training/$TID/status")
say "     final status: $(cat "$RESP")"

say "== 5. curve has real data (learner hook + evaluator) =="
_=$(req GET "/api/training/$TID/curve")
if "$PY" - "$RESP" <<'PYEOF'
import json, sys
c = json.load(open(sys.argv[1], encoding="utf-8"))
assert isinstance(c.get("iterations"), list) and c["iterations"], "iterations missing/empty"
assert isinstance(c.get("eval_rewards"), list) and c["eval_rewards"], "eval_rewards missing/empty"
assert isinstance(c.get("losses"), list) and c["losses"], "losses missing/empty"
assert len(c["iterations"]) == len(c["eval_rewards"]) == len(c["losses"]), "curve arrays misaligned"
assert any(l != 0.0 for l in c["losses"]), "no real (non-zero) loss recorded by the hook"
print("OK rows={} last_iter={} last_reward={} last_loss={}".format(
    len(c["iterations"]), c["iterations"][-1], c["eval_rewards"][-1], c["losses"][-1]))
PYEOF
then
  ok "curve.json valid"
else
  bad "curve.json invalid: $(cat "$RESP")"
fi

say "== 6. model.pth download =="
expect "GET /model       -> 200" 200 "$(req GET "/api/training/$TID/model")"
curl -s -o "$MODELDL" --noproxy "localhost,127.0.0.1" "$BASE/api/training/$TID/model"
SIZE=$(stat -c%s "$MODELDL" 2>/dev/null || echo 0)
if [ "${SIZE:-0}" -gt 50000 ]; then
  ok "model.pth downloaded (${SIZE} bytes)"
else
  bad "model.pth suspiciously small (${SIZE} bytes)"
fi

say "== 7. concurrency queue limit 1 =="
make_body testdata/sample_model.py ppo cartpole 100000
expect "second start     -> 200" 200 "$(req POST /api/training/start "$BODY")"
TID2=$(jget task_id)
say "     task_id=$TID2"
_=$(req GET "/api/training/$TID2/status")
expect "status is running" running "$(jget status)"
make_body testdata/sample_model.py ppo cartpole 100000
expect "third start busy -> 409" 409 "$(req POST /api/training/start "$BODY")"

say "== 8. stop =="
expect "stop             -> 200" 200 "$(req POST "/api/training/$TID2/stop")"
expect "stopped == true" true "$(jget stopped)"
_=$(req GET "/api/training/$TID2/status")
expect "status -> done" done "$(jget status)"
_=$(req GET "/api/training/$TID2/status")
expect "message mentions stop" "stopped by user" "$(jget message)"
expect "model of stopped task -> 404" 404 "$(req GET "/api/training/$TID2/model")"
expect "stop again (idempotent) -> 200" 200 "$(req POST "/api/training/$TID2/stop")"
_=$(req GET "/api/training/$TID2/status")
expect "status still done" done "$(jget status)"

say "=============================================================="
say "RESULT: PASS=$PASS FAIL=$FAIL"
rm -rf "$TMP"
[ "$FAIL" -eq 0 ] && exit 0 || exit 1
