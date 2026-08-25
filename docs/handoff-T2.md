# T2 后端 DI-engine 训练服务 —— 工作总结与交接（handoff-T2）

> 分支：`feature/backend-service`（worktree `D:/WorkSpace/dlp-T2`）
> 契约：`docs/taskcard-v2.5.md` API 契约 v2.2（在 main 分支：`git show origin/main:docs/taskcard-v2.5.md`）
> 环境事实：宿主机 dlbackend（py3.11）无 DI-engine；DI-engine 在 `ding_env`（py3.10），版本 **v0.5.3**，torch 2.12.1，gym 0.25.1 + gymnasium 1.3.0。

---

## 1. 交付物清单（全部在 `backend/` 下，外加本文档）

| 文件 | 说明 |
|------|------|
| `backend/training_service.py` | dlbackend 侧 FastAPI 服务：5 个 API + 白名单校验 + 并发队列限 1 + 子进程管理 |
| `backend/training_worker.py` | 由 `$DING_PYTHON` 运行的训练进程：exec 加载模型 + 三模式自检 + PPO 训练 + hook 写 curve + 存 model.pth |
| `backend/test_training.sh` | Git Bash 版全链路验收脚本（curl） |
| `backend/test_training.ps1` | PowerShell 版全链路验收脚本（curl.exe） |
| `backend/testdata/sample_model.py` | 测试 fixture：符合契约 v2.2 的合规模型（critic 已 squeeze） |
| `backend/testdata/bad_model.py` | 测试 fixture：critic 未 squeeze 返回 (B,1)，用于验证自检 400 |
| `backend/.gitignore` | 忽略 `trained_models/` 运行时产物 |
| `backend/runner.py`（+4 行） | 挂载 training router 到主应用（`app.include_router(training_router)`） |

> 注：`testdata/` 下的 fixture 是 **T2 服务验收用**，与 T3 的 `frontend/src/utils/__tests__/fixtures/sample_model.py`（生成器产物验证）是两回事，不要混淆。

---

## 2. 架构：runner → worker 子进程 + JSON 文件交换

```
POST /api/training/start
  → training_service（dlbackend, py3.11）
      1. 白名单校验（ppo + cartpole）→ 400
      2. 队列检查（并发 ≤ 1）→ 409
      3. 写 backend/trained_models/<task_id>/config.json（含 model_code 原文）
      4. 以 $DING_PYTHON（默认 D:\anaconda3\envs\ding_env\python.exe）起子进程：
           $DING_PYTHON training_worker.py <config.json>
      5. 等待 dummy 自检结论：status.json 变 running → 200 {task_id}；
         变 error → 400 带提示；超时 90s → 先返回 task_id（前端轮询）
  → 状态经文件交换：
       status.json  {"status": starting|running|done|error, "train_iter", "eval_reward", "message"}
       curve.json   {"iterations": [...], "eval_rewards": [...], "losses": [...]}
       model.pth    （仅训练成功完成时保存）
       worker.pid / worker.log
```

设计要点：

- **首帧竞态**：status.json 不存在 → 返回 `{"status":"starting","train_iter":0,"eval_reward":null,"message":""}`；curve.json 不存在 → 空数组。未知 task_id 四个读端点统一 404。
- **自检 400 的时序**：任务卡要求"维度不匹配返回 400"，因此 `/start` 会**阻塞等待自检结论**（最长 90s，torch/ding import 各约 12s）。worker 在最早期（任何重 import 之前）就写 status='starting' 和 worker.pid，尽量缩小"无状态文件"的窗口。
- **并发锁**：`threading.Lock` 包住「校验 + 队列检查 + spawn + 等自检」整段，避免两个请求同时通过队列检查。
- **stop 语义**：`Popen.terminate()`（Windows = TerminateProcess）；ding 的 `BaseEnvManager` 是**进程内**实例化 env（源码 `_create_state: self._envs = [e() for e in self._env_fn]`），杀掉 worker 无孤儿进程。stop 后 status 落 `done` + `"stopped by user"`（契约枚举外的 stopped 状态不用）；幂等，重复 stop 不再改写。stop 不产 model.pth（契约：训练未完成 → GET /model 404）。
- **进程活性**：Windows 上 `os.kill(pid, 0)` 会真的 TerminateProcess，改用 `ctypes` + `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` + `GetExitCodeProcess == STILL_ACTIVE`。status 为 starting/running 但 pid 已死 → 服务把状态纠正为 error（"training worker exited unexpectedly"），覆盖服务重启后句柄丢失的场景。

## 3. training_worker.py 关键实现

1. **模型加载**：`exec(model_code, ns)` 原样执行（约束：只 exec 不修改），ns 预置 `torch/nn/MODEL_REGISTRY`。要求用户代码 `@MODEL_REGISTRY.register('user_net')`；未注册立即报错。
2. **三模式 dummy 自检**（契约 v2.2 钉死）：
   - `compute_actor` → `logit` 形状 `(B, action_shape)`
   - `compute_critic` → `value` 形状 `(B,)`（必须 squeeze，`(B,1)` 直接判失败——这正是省掉"value loss 广播怪错"的关键）
   - `compute_actor_critic` → 两 key 同上述形状
   - 不符 → status='error' + 可读提示，exit 2，`/start` 转成 HTTP 400。
3. **配置**：照抄 `dizoo/classic_control/cartpole/config/cartpole_ppo_config.py` 结构，显式 `model=dict(type='user_net', obs_shape=..., action_shape=..., action_space='discrete')`（不依赖 PPO 内部默认 'vac'）；`create_config.env.manager = EasyDict(type='base')`；collector_env_num=4；`stop_value=195`（eval_reward ≥ 195 提前停）；batch_size/learning_rate/max_train_iter 来自请求 hyperparams。
4. **curve 数据（hook 机制）**——这是任务卡点名"先读源码再动手"的部分，结论如下：
   - **losses 走官方 learner hook**：`register_learner_hook('curve_log', _CurveLogHook)`（`ding/worker/learner/learner_hook.py` 的公开机制），配置注入 `policy.learn.learner.hook.curve_log = dict(type='curve_log', position='after_iter', priority=10)`。priority 10 < log_show 的 20，保证在 LogShowHook 清空 log_buffer **之前**取到 `total_loss`（LogDict 按键覆盖、每 iter 更新，语义正确）。
   - **eval_reward 无法走 hook**：通读 `interaction_serial_evaluator.py` 确认 ding v0.5.3 的 evaluator **没有任何 hook 扩展点**，eval 数据只进 tb_logger 和返回值的 `eval_info`。因此训练主循环**照抄 `serial_entry_onpolicy.py` 的循环体**（同样的 compile_config/get_vec_env_setting/create_policy/BaseLearner/create_serial_collector/InteractionSerialEvaluator/BaseSerialCommander 公开 API，未编造任何接口），在每次 `evaluator.eval()` 返回后取 `eval_info['eval_episode_return']` 均值，与 hook 攒下的窗口内平均 loss 合成一行写入 curve.json。
5. **model.pth**：训练完成（含 195 提前停）后 `torch.save(policy.learn_mode.state_dict(), 'model.pth')`——与 DI-engine 自身 ckpt 同构（含 value_norm wrapper 参数）。
6. **退出码**：0=done，2=自检失败，3=训练异常；异常时 status='error' + traceback 进 worker.log。

## 4. 实测验证结果

- worker 冒烟：坏模型 exit 2（提示 `compute_critic value shape (4, 1) != expected (4,)`）；好模型 250 迭代：curve 两行 `{iter:0, reward:9.0, loss:0.0}`、`{iter:100, reward:200.0, loss:1.4175}`——hook 采到真实 loss。
- 真实训练（seed=0，样本 MLP）：100 迭代 eval_reward 9.0 → 200.0，触发 stop_value=195 提前停止，status `done (eval_reward reached stop_value)`，model.pth 122KB。
- **`test_training.sh` 与 `test_training.ps1` 均 22/22 PASS**，覆盖：四端点未知 id 404、白名单 400（sac/pendulum）、坏模型自检 400、真实训练到 done、curve 数据校验（含非零 loss）、model.pth 下载、队列 409、stop 全语义（含幂等）。

## 5. 遇到的问题与解决方法

1. **任务卡参考路径与 ding v0.5.3 实际布局不符**（最重要的一组）：
   - `MODEL_REGISTRY` 在 `ding.utils`，不在 `ding.model`（v0.5.3 的 `ding/model/__init__.py` 不导出它）；
   - 没有 `ding/policy/command_mode/ppo_command.py`，`'ppo'/'ppo_command'` 是**导入 `ding.policy` 时才注册**的（`command_mode_policy_instance.py`），只 import `ding.utils` 查注册表会误判为不存在；
   - `design_pattern_helper.py`、`compile_config.py`（在 `ding/config/config.py` 内）等路径均不存在。
   - **解决**：所有结论都以实际解释器探针（import + inspect.signature + grep 源码）验证为准，不照搬旧版文档。
2. **evaluator 无 hook**：见 §3.4。**解决**：learner hook 负责 loss，重抄 onpolicy 主循环取 eval_info，两路数据合成 curve.json。
3. **Windows 进程控制**：`os.kill(pid, 0)` 会 TerminateProcess → 用 ctypes OpenProcess 判定活性；`Popen(..., creationflags=CREATE_NO_WINDOW)` 避免服务模式下弹控制台窗口。
4. **Git Bash /tmp 与 Windows python 路径不一致**：冒烟测试时 bash 写 `/tmp/...`、dlbackend python 按 `D:\tmp\...` 解析 → FileNotFoundError。**解决**：用 `tempfile.mkdtemp()` 生成 Windows 原生路径。
5. **worker 内 torch 局部 import 不生效**：`main()` 里局部 `import torch` 不会进模块全局命名空间，`dummy_forward_selfcheck` 引用 `torch` 会 NameError。**解决**：函数内各自 import（同时把重 import 推迟到写完首帧 status/pid 之后，压缩"无状态"窗口）。
6. **PowerShell 5.1 的两个坑**（ps1 脚本首跑失败的原因）：
   - `ConvertTo-Json` 把**含换行的字符串**序列化成 `{"value": "..."}` → 后端收到 model_code 为对象 → 422。**解决**：请求体改手写 JSON 字符串（schema 固定、可控），配自写转义函数。
   - `Get-Content` 默认按 ANSI(GBK) 读 UTF-8 文件 → fixture 里 `——` 乱码。**解决**：`-Encoding UTF8` 读写，输出用 `[IO.File]::WriteAllText(..., UTF8Encoding($false))` 避免 BOM 被 `--data-binary` 带进请求体。
   - 另：`"$desc: ..."` 中 `$desc:` 被 PS 解析成驱动器变量 → 改用 `${desc}`。
7. **hook 配置注入路径的确认**：`compile_config` 里 `policy_config_template` 预置了 `learn.learner=dict()`，用户配置经 `deep_merge_dicts`（递归 `deep_update`）合并，自定义 hook 条目能正确落到 `learner.hook`——这是 cartpole 配置没有 `learn.learner` 却能跑的原因，也是 hook 配置能工作的依据。
8. **start 阻塞时长**：torch import ~12s + ding.policy import ~12s，自检等待上限设 90s；超时降级为返回 task_id + 前端轮询（契约的 starting 帧兜底）。

## 6. 遗留问题与风险

1. **未 commit/push**：所有改动在工作区（`git status` 仅 backend/ 相关文件 + 本文档）。合并顺序按任务卡：codegen-diengine → backend-service → frontend-panel；`runner.py` 的 4 行挂载与 T1 无冲突。
2. **服务重启后的 stop**：内存里的 Popen 句柄丢失，靠 status.json 里的 worker.pid + `taskkill /F /T` 兜底；训练进行中服务崩溃再重启，状态自愈依赖 pid 活性检查（worker 死了会被纠正成 error）。
3. **队列占用的判定是启发式**：worker 解释器启动的 1~2s 内既无 status 也无 pid，用"目录/status 文件 mtime < 90s 视为占用"兜底；极端情况下刚崩溃的任务会多占 90s 队列。
4. **`/start` 阻塞最多 90s**：前端首个请求体验较慢（自检必须同步返回 400 所致）；超时路径下坏模型要到轮询阶段才能看到 error 而非 400。
5. **model.pth 结构**：保存的是 `learn_mode.state_dict()`（含 value_norm 等 wrapper 参数），与 DI-engine ckpt 同构；下游若想直接 `model.load_state_dict()` 到裸用户模型需先剥 wrapper（MVP 只要求下载交付，不阻塞）。
6. **训练产物无自动清理**：`backend/trained_models/<task_id>/` 会持续累积（含失败的 status=error 目录）；MVP 未做 TTL/清理接口。
7. **固定参数**：seed=0、eval_freq=100、collector_env_num=4 写死在 worker 配置里；契约仅要求 hyperparams 暴露 lr/batch/max_train_iter。
8. **测试脚本对"收敛"有假设**：`.sh`/`.ps1` 第 5 步断言 `max(eval_rewards) >= 195`。固定 seed + 该 MLP fixture 下必然收敛（已反复实测 100 iter 到 200），但换成不收敛的模型结构时该断言会红——这是脚本预期，不是服务缺陷。
9. **worker 崩溃信息截断**：`/start` 的 500 详情只带 worker.log 尾部 2000 字符。
10. **SAC/连续动作**：契约明确 MVP 之外不实现，无遗留代码。

## 7. 运行指南

```bash
# 后端（dlbackend 环境；无需 Docker）
conda activate dlbackend
cd backend
uvicorn training_service:app --host 0.0.0.0 --port 8000   # 或 runner:app（含既有 torchlens 路由）

# 验收测试（任选其一，均 22 项断言）
bash test_training.sh
powershell -NoProfile -ExecutionPolicy Bypass -File .\test_training.ps1

# 环境变量
DING_PYTHON   # 默认 D:\anaconda3\envs\ding_env\python.exe（服务端读，勿硬编码）
TRAINING_BASE # 测试脚本用，默认 http://localhost:8000
PYTHON        # 仅 .sh 的 JSON 辅助用，默认 python / dlbackend python
```

## 8. 与契约 v2.2 的对应关系（自查）

- 5 个 API + 首帧竞态语义 + 未知 id 统一 404 ✓（两脚本实测）
- 白名单仅 ppo+cartpole，越界 400 ✓
- 并发队列限 1 ✓（409 + 锁防竞态）
- DING_PYTHON 环境变量，默认 ding_env 路径 ✓
- exec 加载 + `@MODEL_REGISTRY.register('user_net')` + 显式 `model=dict(type='user_net')` ✓
- Windows 兼容：env.manager=base、collector_env_num=4 ✓
- dummy 自检三模式、形状不符立即 400 ✓
- curve 用 hook 机制（先读源码确认，未编造 API）✓
- 训练结束保存 model.pth ✓；无 model 时 GET /model 404 ✓
- 约束：只改 backend/（+本文档）；model_code 只 exec 不修改 ✓
