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
#   powershell -ExecutionPolicy Bypass -File .\test_training.ps1
#
# Env override: $env:TRAINING_BASE (default http://localhost:8000).
# =============================================================================
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

$base = if ($env:TRAINING_BASE) { $env:TRAINING_BASE } else { "http://localhost:8000" }
$tmp  = Join-Path $env:TEMP ("t2test_" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Path $tmp | Out-Null
$resp     = Join-Path $tmp "resp.json"
$body     = Join-Path $tmp "body.json"
$modelDl  = Join-Path $tmp "downloaded.pth"

$script:PASS = 0
$script:FAIL = 0
function Ok($msg)  { $script:PASS++; Write-Host "  [PASS] $msg" }
function Bad($msg) { $script:FAIL++; Write-Host "  [FAIL] $msg" }
function Expect($desc, $want, $got) {
    if ("$got" -eq "$want") { Ok "$desc ($got)" } else { Bad "${desc}: wanted $want got $got" }
}

# Invoke-Api <method> <path> [bodyfile] -> writes $resp, returns HTTP status code
function Invoke-Api($method, $path, $bodyFile) {
    if ($bodyFile) {
        return curl.exe -s -o $resp -w "%{http_code}" -X $method "$base$path" `
            -H "Content-Type: application/json" --data-binary "@$bodyFile"
    }
    return curl.exe -s -o $resp -w "%{http_code}" -X $method "$base$path"
}

function Get-RespJson { Get-Content -Raw -Encoding UTF8 $resp | ConvertFrom-Json }

# New-StartBody <modelfile> <algorithm> <env> <maxTrainIter> -> writes $body
# NOTE: built by hand instead of ConvertTo-Json because Windows PowerShell 5.1
# serializes multi-line strings as {"value": "..."}, which breaks the API.
function ConvertTo-JsonEscaped([string]$s) {
    $s = $s -replace '\\', '\\'
    $s = $s -replace '"', '\"'
    $s = $s -replace "`r`n", '\n'
    $s = $s -replace "`n", '\n'
    $s = $s -replace "`r", ''
    $s = $s -replace "`t", '\t'
    return $s
}
function New-StartBody($modelFile, $algo, $envName, $iters) {
    $code = Get-Content -Raw -Encoding UTF8 $modelFile
    $json = '{"network":{"model_code":"' + (ConvertTo-JsonEscaped $code) + '","obs_shape":4,"action_shape":2},' +
            '"algorithm":"' + $algo + '","env":"' + $envName + '",' +
            '"hyperparams":{"learning_rate":0.001,"batch_size":256,"max_train_iter":' + $iters + '}}'
    [System.IO.File]::WriteAllText($body, $json, (New-Object System.Text.UTF8Encoding($false)))
}

# Wait-Status <taskId> <want> <timeoutSec> -> $true on match
function Wait-Status($tid, $want, $timeoutSec = 900) {
    $t0 = Get-Date
    while ($true) {
        $null = Invoke-Api GET "/api/training/$tid/status"
        $st = (Get-RespJson).status
        if ($st -eq $want) { return $true }
        if ($st -eq "error") { Write-Host "     status=error message: $((Get-RespJson).message)"; return $false }
        if (((Get-Date) - $t0).TotalSeconds -ge $timeoutSec) { return $false }
        Start-Sleep -Seconds 2
    }
}

Write-Host "== 1. server up + unknown task_id -> 404 on all four endpoints =="
Expect "GET  /status unknown" 404 (Invoke-Api GET "/api/training/no_such_task/status")
Expect "POST /stop   unknown" 404 (Invoke-Api POST "/api/training/no_such_task/stop")
Expect "GET  /curve  unknown" 404 (Invoke-Api GET "/api/training/no_such_task/curve")
Expect "GET  /model  unknown" 404 (Invoke-Api GET "/api/training/no_such_task/model")

Write-Host "== 2. MVP whitelist: algorithm/env outside [ppo, cartpole] -> 400 =="
New-StartBody "testdata\sample_model.py" sac cartpole 100
Expect "algorithm=sac    -> 400" 400 (Invoke-Api POST "/api/training/start" $body)
New-StartBody "testdata\sample_model.py" ppo pendulum 100
Expect "env=pendulum     -> 400" 400 (Invoke-Api POST "/api/training/start" $body)

Write-Host "== 3. dummy-forward self-check: critic returns (B,1) -> 400 on /start =="
New-StartBody "testdata\bad_model.py" ppo cartpole 100
Expect "bad critic shape -> 400" 400 (Invoke-Api POST "/api/training/start" $body)
Write-Host "     detail: $((Get-RespJson).detail)"

Write-Host "== 4. real training: ppo + cartpole, max_train_iter=3000 =="
New-StartBody "testdata\sample_model.py" ppo cartpole 3000
Expect "start            -> 200" 200 (Invoke-Api POST "/api/training/start" $body)
$tid = (Get-RespJson).task_id
Write-Host "     task_id=$tid"
if (Wait-Status $tid "done" 900) { Ok "status reached 'done'" } else { Bad "status never reached 'done'" }
Write-Host "     final status: $(Get-Content -Raw $resp)"

Write-Host "== 5. curve has real data (learner hook + evaluator) =="
$null = Invoke-Api GET "/api/training/$tid/curve"
$curve = Get-RespJson
$okCurve = $true
if (-not $curve.iterations -or @($curve.iterations).Count -eq 0) { $okCurve = $false }
if (-not $curve.eval_rewards -or @($curve.eval_rewards).Count -eq 0) { $okCurve = $false }
if (-not $curve.losses -or @($curve.losses).Count -eq 0) { $okCurve = $false }
if (@($curve.iterations).Count -ne @($curve.losses).Count) { $okCurve = $false }
if (@($curve.losses | Where-Object { $_ -ne 0 }).Count -eq 0) { $okCurve = $false }
if ($okCurve) {
    Ok "curve.json valid (rows=$(@($curve.iterations).Count) last_reward=$($curve.eval_rewards[-1]) last_loss=$($curve.losses[-1]))"
} else {
    Bad "curve.json invalid: $(Get-Content -Raw $resp)"
}

Write-Host "== 6. model.pth download =="
Expect "GET /model       -> 200" 200 (Invoke-Api GET "/api/training/$tid/model")
curl.exe -s -o $modelDl "$base/api/training/$tid/model"
$size = (Get-Item $modelDl).Length
if ($size -gt 50000) { Ok "model.pth downloaded ($size bytes)" } else { Bad "model.pth suspiciously small ($size bytes)" }

Write-Host "== 7. concurrency queue limit 1 =="
New-StartBody "testdata\sample_model.py" ppo cartpole 100000
Expect "second start     -> 200" 200 (Invoke-Api POST "/api/training/start" $body)
$tid2 = (Get-RespJson).task_id
Write-Host "     task_id=$tid2"
$null = Invoke-Api GET "/api/training/$tid2/status"
Expect "status is running" running (Get-RespJson).status
New-StartBody "testdata\sample_model.py" ppo cartpole 100000
Expect "third start busy -> 409" 409 (Invoke-Api POST "/api/training/start" $body)

Write-Host "== 8. stop =="
Expect "stop             -> 200" 200 (Invoke-Api POST "/api/training/$tid2/stop")
Expect "stopped == true" true (Get-RespJson).stopped
$null = Invoke-Api GET "/api/training/$tid2/status"
Expect "status -> done" done (Get-RespJson).status
$null = Invoke-Api GET "/api/training/$tid2/status"
Expect "message mentions stop" "stopped by user" (Get-RespJson).message
Expect "model of stopped task -> 404" 404 (Invoke-Api GET "/api/training/$tid2/model")
Expect "stop again (idempotent) -> 200" 200 (Invoke-Api POST "/api/training/$tid2/stop")
$null = Invoke-Api GET "/api/training/$tid2/status"
Expect "status still done" done (Get-RespJson).status

Write-Host "=============================================================="
Write-Host "RESULT: PASS=$($script:PASS) FAIL=$($script:FAIL)"
Remove-Item -Recurse -Force $tmp
if ($script:FAIL -eq 0) { exit 0 } else { exit 1 }
