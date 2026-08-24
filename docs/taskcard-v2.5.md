# DL-Playground × DI-engine 融合 MVP —— 任务卡 v2.5（最终定稿）

> 目标：fork DL-Playground，改造为"网页拖拽搭网络 + 选择 RL 算法训练"的平台（对标 Utenet-Nash 最小验证版）
> 仓库：https://github.com/dsgiitr/DL-Playground （React + TypeScript 前端，Python 后端）
> v2.5 变更：T3 加 vitest 唯一例外（仓库无测试基础设施，已核验 package.json）、T0 加 docs/ 提交任务卡步骤（代理可读契约全文）
> v2.4 变更：两层测试接通水管（fixture JSON + node 写盘 sample_model.py + Python 读盘 exec）、curve 首帧返回空数组、未知 task_id 四端点统一 404、SAC 版本号措辞统一
> v2.3 变更：T3 测试拆两层（node 字符串断言 + $DING_PYTHON 形状断言）、契约版本号统一 v2.2、T1 补 nodes/edges 来源指针、默认 max_train_iter 5000（演示友好 + eval_reward≥195 提前停）、status 首帧竞态处理（starting 状态）
> v2.2 变更：T3 参照改为 vac.py（cartpole 无专属 model，已实测确认）、返回结构三行钉进契约（含 (B,) 细节）、加 compute_actor_critic 模式、T1 手写 SVG 折线图、T2 显式 model type + dummy 自检测三模式

---

## 分支与工作目录约定

| 任务 | 代理 | 模型 | 分支 | 工作目录（git worktree） | 涉及目录 |
|:--:|:--:|------|------|------|----------|
| T0 | Claude Code | qwen3.8-max / deepseek-v4-flash | main | `<repo>`（主 checkout） | 全仓库（只做收尾） |
| T1 | OpenCode | kimi-k2.7-code | feature/frontend-panel | `<repo>/../dlp-T1` | frontend/src/ |
| T2 | OpenCode | kimi-k2.7-code | feature/backend-service | `<repo>/../dlp-T2` | backend/ |
| T3 | OpenCode | glm-5.2 | feature/codegen-diengine | `<repo>/../dlp-T3` | frontend/src/utils/ |

**工作目录隔离（强制）**：T1/T2/T3 **绝不在同一份 checkout 上并发**，每个任务用独立 git worktree：

```bash
# 在 T0 完成后执行（main 上）：
git worktree add ../dlp-T1 feature/frontend-panel
git worktree add ../dlp-T2 feature/backend-service
git worktree add ../dlp-T3 feature/codegen-diengine
```

**约定**：各代理的 cwd 指向自己的 worktree 目录；跨任务依赖一律通过 API 契约对接；合并由人工在 main 执行。

---

## 模型与执行配置

### 工具接入说明

| 工具 | 接入方式 | 说明 |
|------|----------|------|
| **Claude Code** | DeepSeek Anthropic 端点 或 阿里 Token Plan | 阿里：百炼 Anthropic 端点 + qwen3.8-max；DeepSeek：`https://api.deepseek.com/anthropic` + v4-flash |
| **OpenCode** | OpenCode Go 套餐 | T1/T2 用 `kimi-k2.7-code`（额度 6,750/月，最足），T3 用 `glm-5.2`（1M 上下文读生成器源码） |
| **Codex**（附加，代码审查用） | DeepSeek Responses API | 官方一键脚本，仅 v4-flash 支持；`codex review --base origin/main` |

### 额度注意事项 ⚠️

- OpenCode Go 每月 60 美元额度：kimi-k2.7-code 最足（重活用它）；glm-5.2 约 4,300；kimi-k3 仅 ~490（MVP 不用）
- DeepSeek 官方 key 极便宜，高频简单任务用它
- T2 是工作量重心，额度留足（k2.7-code）

### 执行命令

```bash
# T0（Claude Code，主 checkout）
claude -p "【T0任务描述】" --allowedTools "Read,Write,Bash,Edit" --max-turns 20

# T1 / T2 / T3（OpenCode，各自 worktree）
cd ../dlp-T1 && opencode run "【T1任务描述】" --model kimi-k2.7-code
cd ../dlp-T2 && opencode run "【T2任务描述】" --model kimi-k2.7-code
cd ../dlp-T3 && opencode run "【T3任务描述】" --model glm-5.2
```

---

## API 契约 v2.2（前端 ↔ 后端，T1/T2/T3 共同遵守）

### HTTP 接口

```
POST /api/training/start
  Request: {
    "network": {
      "model_code": "...",          # ← 必须是 T3 桩文件/generateDiEngineModel 的产出（DI-engine 合规类）
      "obs_shape": 18,
      "action_shape": 5
    },
    "algorithm": "ppo",             # MVP 仅支持: ppo（SAC 移入 v2.2+ / MVP 之后，不实现）
    "env": "cartpole",              # MVP 仅支持: cartpole
    "hyperparams": {                # 可选，缺省用默认
      "learning_rate": 0.001,
      "batch_size": 256,
      "max_train_iter": 5000        # ⚠️ 演示友好默认值！100000 在 CPU 上要几小时，没人等得到 done。
    }                               #     CartPole PPO 几百~几千迭代即收敛（eval_reward 接近 200），
                                    #     建议默认 5000~10000；训练中 eval_reward ≥ 195 可提前停止
  }
  Response: { "task_id": "abc123" }

GET /api/training/{task_id}/status
  Response: {
    "status": "starting" | "running" | "done" | "error",   # ⚠️ 首帧竞态：worker 尚未写出第一个 status.json 时
    "train_iter": 0,                                        #     返回 {"status":"starting","train_iter":0}，不要 404/500
    "eval_reward": null,
    "message": "..."
  }

POST /api/training/{task_id}/stop
  Response: { "stopped": true }

GET /api/training/{task_id}/curve
  Response: {
    "iterations": [0, 500, 1000, ...],
    "eval_rewards": [-90.1, -88.5, ...],
    "losses": [24.5, 23.9, ...]
  }
  # ⚠️ 首帧竞态同 status：curve.json 未生成时返回空数组
  #    {"iterations":[],"eval_rewards":[],"losses":[]}，不要 404
  #    （T1 的 SVG 图对空数组要能渲染"暂无数据"）

GET /api/training/{task_id}/model
  Response: 模型文件下载（.pth）
  （训练未完成/无 model.pth 时返回 404 + 提示文案）

# ⚠️ 统一规则：status/stop/curve/model 四个端点对【不存在的 task_id】
#    一律返回 404（行为一致），只有"存在但暂无数据"才返回空值/starting
```

### TS 侧契约（T0 落桩，T1 调用，T3 实现）

```typescript
// T0 在 main 上创建 frontend/src/utils/codeCompileDiEngine.ts（桩文件，只含签名）：
export function generateDiEngineModel(
  nodes: Node[], edges: Edge[]
): { model_code: string; obs_shape: number; action_shape: number } {
  throw new Error("not implemented, see T3");
}

// T3 在该文件填入真实实现（不得改签名）
// T1 的训练提交流程调用此函数，返回值塞进 POST /api/training/start 的 network 字段
```

**契约要点**：
- `model_code` 唯一来源 = `generateDiEngineModel()`；T1 不得调用旧生成器 codeCompile.ts
- T2 只负责 exec 加载 + MODEL_REGISTRY 注册 + 训练，不修改网络结构
- algorithm/env 超出 MVP 白名单（仅 ppo + cartpole）→ 返回 400 + 提示文案

### 模型返回结构契约（v2.2 钉死，T3 必须严格实现，已实测 vac.py 确认）

```python
# T3 生成的模型类 forward(x, mode) 返回结构（参照 ding/model/template/vac.py，注册名 'vac'）：
#   'compute_actor'        → {'logit': Tensor (B, action_shape)}   # 离散 logits，无 tanh
#   'compute_critic'       → {'value':  Tensor (B,)}               # ⚠️ 一维！必须 squeeze 掉最后一维
#   'compute_actor_critic' → {'logit': ..., 'value': ...}          # 两个 key 同时返回
#
# ⚠️ (B,) 细节：critic 头输出必须 squeeze，若输出 (B,1)，
#    PPO 的 value loss 广播时会出怪错——这是省代理好几轮调试的关键
# ⚠️ dizoo/classic_control/cartpole/ 下没有 model 目录（已实测），离散模板就是 vac.py
```

---

## 参考路径总览

> ⚠️ 环境路径注意：本机有两个 Anaconda！**ding_env 在 `D:\anaconda3\envs\ding_env`（DI-engine 所在地）**；`D:\AI_LEARNING\Anaconda3` 是另一个独立安装。代理开工第一步先 `conda env list` 确认真实路径。**ding python 路径不要硬编码，用环境变量 `DING_PYTHON`（默认 `D:\anaconda3\envs\ding_env\python.exe`）。**

```
# DI-engine 关键源码（T2/T3 用）
D:\anaconda3\envs\ding_env\lib\site-packages\ding\entry\serial_entry.py              ← serial_pipeline 训练入口 + hook 扩展点（T2 必读）
D:\anaconda3\envs\ding_env\lib\site-packages\ding\utils\registry.py                  ← MODEL_REGISTRY 注册机制
D:\anaconda3\envs\ding_env\lib\site-packages\ding\model\template\qac.py              ← 连续模型模板（仅参考，勿照抄）

# dizoo 配置（T2 抄结构）
D:\anaconda3\envs\ding_env\lib\site-packages\dizoo\classic_control\cartpole\config\cartpole_ppo_config.py   ← PPO 配置（MVP 算法）

# 模型模板（T3 关键参照——注意 cartpole 无专属 model 文件，离散模板就是 vac.py）
D:\anaconda3\envs\ding_env\lib\site-packages\ding\model\template\vac.py   ← 离散 actor-critic 模板（注册名 'vac'），返回结构见"模型返回结构契约"
D:\anaconda3\envs\ding_env\lib\site-packages\ding\model\template\qac.py   ← 连续模板（仅了解差异，勿照抄）

# DL-Playground 前端关键文件（T1/T3 用）
frontend/src/features/editor/components/EditorSidebar.tsx      ← 现有侧边栏（新面板落点）
frontend/src/utils/codeCompile.ts                              ← 旧生成器（T3 参考，勿修改）
frontend/src/utils/codeCompileDiEngine.ts                      ← 桩文件（T0 创建，T3 实现）
frontend/src/node_gen/CreateNodeComponent.tsx                  ← 节点定义体系
```

---

## T0：仓库收尾 + 桩文件（Claude Code）—— 精简版

**背景**：环境搭建已完成（前端依赖已装、Dockerfile 已改 CPU 版、dlbackend 就绪）。**不重跑环境搭建。**

**任务描述**：
1. 提交未提交修改（backend/Dockerfile、frontend/package-lock.json、frontend/src/utils/traceService.ts）：
   ```bash
   git add -A && git commit -m "chore: local dev setup (CPU torch image, localhost API)"
   ```
2. 确认/创建团队 fork：检查 `git remote -v`；origin 若仍指上游 dsgiitr，改为团队 fork（与用户确认）
3. **创建桩文件** `frontend/src/utils/codeCompileDiEngine.ts`（契约落进代码）：
   ```typescript
   import type { Node, Edge } from "@xyflow/react";
   export function generateDiEngineModel(
     nodes: Node[], edges: Edge[]
   ): { model_code: string; obs_shape: number; action_shape: number } {
     throw new Error("not implemented, see T3");
   }
   ```
   提交桩文件（T1 依赖它编译通过）。
4. 创建三个 feature 分支（从 main）：
   ```bash
   git checkout -b feature/frontend-panel main
   git checkout -b feature/backend-service main
   git checkout -b feature/codegen-diengine main
   ```
5. **创建三个 git worktree**（T1/T2/T3 独立目录，防打架）：
   ```bash
   git worktree add ../dlp-T1 feature/frontend-panel
   git worktree add ../dlp-T2 feature/backend-service
   git worktree add ../dlp-T3 feature/codegen-diengine
   ```
6. 更新 README：补"本地启动说明"（前端 npm run dev；后端 dlbackend + uvicorn；Docker 镜像构建；**DING_PYTHON 环境变量说明**）
7. **把任务卡 v2.4 commit 进仓库**：复制到 `docs/taskcard-v2.4.md` 并提交——让 T1/T2/T3 代理开工时能读到契约全文（现只能靠 prompt 粘贴；进仓库后 `codex review --base origin/main` 也能对照契约审）

**验收标准**：
- [ ] 本地修改已 commit
- [ ] remote 指向团队 fork（或用户确认保持）
- [ ] 桩文件存在且含契约签名
- [ ] 三个 feature 分支 + 三个 worktree 存在
- [ ] README 含本地启动说明
- [ ] 任务卡 v2.4 已 commit 到 docs/

**约束**：不改功能代码。

---

## T1：前端训练面板（OpenCode + kimi-k2.7-code，worktree: ../dlp-T1）

**目标**：新增"训练配置与监控面板"。

**参考文件**：
```
frontend/src/features/editor/components/EditorSidebar.tsx  ← 侧边栏（面板落点，风格对齐）
frontend/src/features/editor/hooks/useGraphState.ts        ← ⚠️ 当前图的 nodes/edges 状态在这里取（或看 FlowEditor 里的组合方式），别满仓库找
frontend/src/utils/codeCompileDiEngine.ts                  ← 桩文件（已存在，直接 import 编译）
frontend/package.json                                      ← 已有依赖
```

**任务描述**：
1. 侧边栏新增"训练"面板：
   - **算法下拉：仅 PPO 一个选项**（SAC/IQL/MADDPG 置灰"即将支持"）
   - 环境下拉：仅 CartPole（其余置灰）
   - 超参：学习率、batch_size、max_train_iter（可折叠，默认兜底）
   - 【开始训练】/【停止训练】按钮
2. 开始训练（**必须调桩文件函数，不得调用旧生成器**）：
   ```typescript
   import { generateDiEngineModel } from "../utils/codeCompileDiEngine";
   const { model_code, obs_shape, action_shape } = generateDiEngineModel(nodes, edges);
   // 调 POST /api/training/start，network 字段用返回值
   ```
3. 训练期间：每 2 秒轮询 status；拉 curve **用手写 SVG 折线图**展示（⚠️ 项目当前零图表依赖，**不要引入 recharts**，保持轻量）；【停止】调 POST stop
4. 完成显示"下载模型"按钮
5. 后端未就绪时显示"后端不可用"，不崩溃（可先对 mock 接口开发）

**验收标准**：
- [ ] 面板只有 PPO + CartPole
- [ ] import 桩文件编译通过（**桩文件 throw 在运行时才触发，编译期必须通过**）
- [ ] 开始/停止调对应 API
- [ ] 曲线/状态实时更新
- [ ] 后端不可用优雅降级

**约束**：只改 frontend/src/（除 codeCompileDiEngine.ts——那是 T3 的）；遵循契约 v2.2。

---

## T2：后端 DI-engine 训练服务（OpenCode + kimi-k2.7-code，worktree: ../dlp-T2）

**目标**：后端新增训练服务，接入 DI-engine。**工作量重心。**

**背景（架构约束）**：宿主机 dlbackend（py3.11）无 DI-engine；DI-engine 在 `D:\anaconda3\envs\ding_env`（py3.10）。**不能在后端进程直接 import ding**。

**解决方案（照抄项目 runner→worker 模式）**：
```
POST /api/training/start
  → runner（dlbackend）把 model_code + 配置写 JSON 文件
  → 用 $DING_PYTHON（环境变量，默认 D:\anaconda3\envs\ding_env\python.exe）起子进程：
      $DING_PYTHON training_worker.py <config.json>
  → training_worker.py 内：exec(model_code) → MODEL_REGISTRY 注册 → serial_pipeline 训练
  → 状态通过 JSON 文件交换（status.json / curve.json / model.pth）
```

**任务描述**：
1. `backend/training_service.py`（dlbackend 环境，FastAPI）：
   - POST start：校验白名单（仅 ppo + cartpole，超范围 400）→ 写配置 JSON → 启动子进程 → 返回 task_id
   - GET status / POST stop（杀子进程）/ GET curve / GET model（无 model.pth 时 404 + 提示）
   - 并发队列限制 1
   - **ding python 路径不硬编码**：读环境变量 `DING_PYTHON`（默认 `D:\anaconda3\envs\ding_env\python.exe`）
2. `backend/training_worker.py`（用 $DING_PYTHON 运行）：
   - exec 加载 model_code → `@MODEL_REGISTRY.register('user_net')`
   - 参考 `dizoo/classic_control/cartpole/config/cartpole_ppo_config.py` 组装配置（照抄结构），**显式写 `model=dict(type='user_net', ...)`**——不要学 cartpole 配置省略 type（省略时依赖内部默认行为，显式最稳）
   - Windows 兼容：`create_config.env.manager = EasyDict(type='base')`，collector_env_num=4
   - **训练前 dummy forward 自检**：用 env 真实 obs_shape/action_shape 实例化模型，**跑 compute_actor / compute_critic / compute_actor_critic 三种 mode**，验证返回结构符合"模型返回结构契约"（logit 形状 (B, action_shape)、value 形状 (B,) 一维）；不符 → 立即报错返回 400 带提示（防止维度不匹配训练跑飞）
   - **curve 数据来源（关键）**：serial_pipeline 是黑盒，定期写 curve.json 必须有抓手。**先用 hook 机制**：读 `ding/entry/serial_entry.py` 的 hook 扩展点（learner hook / evaluator），自定义 hook 类定期把 train_iter/eval_reward 写入 JSON。**不许编造 API——先读源码确认 hook 用法再动手**；备选方案：解析 tensorboard tfevents
   - 训练结束保存 model.pth 到 backend/trained_models/{task_id}/
3. **交付 curl 测试脚本**（backend/test_training.sh / .ps1）：验证 start→status→curve→stop→model 全链路

**验收标准**：
- [ ] 5 个 API 可用（curl 可测）
- [ ] curl 提交 cartpole+ppo 真正训练（走 ding_env 子进程）
- [ ] dummy forward 自检：维度不匹配返回 400 提示
- [ ] stop 杀子进程；无 model 时 GET /model 返回 404
- [ ] curve 有真实数据（hook 机制，非编造）
- [ ] 测试脚本通过

**参考文件**：
```
D:\anaconda3\envs\ding_env\lib\site-packages\dizoo\classic_control\cartpole\config\cartpole_ppo_config.py
D:\anaconda3\envs\ding_env\lib\site-packages\ding\entry\serial_entry.py   ← 必读：hook 扩展点
D:\anaconda3\envs\ding_env\lib\site-packages\ding\utils\registry.py
backend/runner.py   ← 现有 runner 模式参考
```

**约束**：只改 backend/；遵循契约 v2.2；model_code 只 exec 不修改。

---

## T3：代码生成器对接 DI-engine 模型接口（OpenCode + glm-5.2，worktree: ../dlp-T3）

**目标**：实现 `frontend/src/utils/codeCompileDiEngine.ts`（T0 桩文件的真实实现）。

**参考文件**：
```
frontend/src/utils/codeCompileDiEngine.ts                  ← 桩文件（签名已定，填实现）
frontend/src/utils/codeCompile.ts                          ← 旧生成器（先读懂，勿修改）
frontend/src/node_gen/BaseClass.tsx                        ← 节点基类
frontend/src/features/editor/hooks/useCodeGeneration.ts    ← 旧生成 hook（了解输入输出）

# ⚠️ 接口参照（关键！MVP 是 PPO + CartPole = 离散动作）：
D:\anaconda3\envs\ding_env\lib\site-packages\ding\model\template\vac.py   ← 离散 actor-critic 模板（注册名 'vac'），按"模型返回结构契约"实现
D:\anaconda3\envs\ding_env\lib\site-packages\ding\model\template\qac.py   ← 仅了解连续模型差异，勿照抄
```

**任务描述**：
1. 读懂 codeCompile.ts 的图→代码翻译逻辑
2. 实现 `generateDiEngineModel(nodes, edges)`（**签名与 T0 桩一致，不得改**）：
   - 生成模型类：含 obs_shape/action_shape 构造参数
   - `forward(x, mode)` 支持**三种模式**（严格按"模型返回结构契约"）：
     - `'compute_actor'` → `{'logit': (B, action_shape)}`（离散 logits，无 tanh）
     - `'compute_critic'` → `{'value': (B,)}`（**一维，squeeze 掉最后一维**）
     - `'compute_actor_critic'` → 两个 key 同时返回（PPO 某些版本 collect 阶段会调，三行成本，必做）
   - **动作类型按算法决定，不要一刀切 tanh**：MVP 只有 PPO+CartPole（离散动作）→ logits 头
3. obs_shape/action_shape 从图结构推导（输入/输出节点维度）
4. MVP 只保证 MLP 结构
5. **测试拆两层 + 水管接通**（⚠️ node 跑不了 torch；且生成器是 TS，Python 拿不到产出字符串——**必须通过 fixture 文件传递**）：
   - **标准样例图 fixture**：定义简单 MLP 的 nodes/edges JSON（`__tests__/fixtures/sample_graph.json`），两层测试共用
   - **node 测试**（`__tests__/diengine_codegen.test.ts`）：跑 `generateDiEngineModel(sample_graph)` → 断言**字符串结构**（三模式分支、key 名 logit/value、critic 头有 squeeze 调用）→ **把返回的 model_code 写盘到 `__tests__/fixtures/sample_model.py`**
   - **Python 测试**（交付物 `__tests__/diengine_model_test.py`）：**读 `fixtures/sample_model.py`**（不是手写样例代码！）→ exec → 实例化 → dummy forward，断言 `logit (B, action_shape)`、`value (B,)` 等形状
   - **运行顺序（写进交付说明）**：先 `npm test`（产出 fixture）→ 再 `$DING_PYTHON diengine_model_test.py`。**严禁 Python 测试手写样例代码**——那就验的不是生成器了
   - 这让 T3 在自己分支就能验证契约（生成器→Python 全链路），不用等 T2

**验收标准**：
- [ ] 函数签名与桩文件一致
- [ ] 生成的模型符合 vac.py 离散接口（logit/value/compute_actor_critic 三模式，value 一维）
- [ ] 简单 MLP 图生成合法代码
- [ ] 旧生成器无回归
- [ ] node 测试通过（字符串结构断言 + 产出 sample_model.py fixture）
- [ ] Python 测试通过（$DING_PYTHON 读 fixture 跑形状断言）

**约束**：只改 frontend/src/utils/codeCompileDiEngine.ts + `__tests__/`（或新增辅助文件）；不破坏旧生成器。
**唯一例外**：仓库当前**没有测试基础设施**（package.json 无 test 脚本、无 vitest/jest，已核验）——**允许修改 frontend/package.json，仅限：① 加 vitest devDependency ② 加 `"test": "vitest run"` 脚本**（测试基础设施，唯一例外；与项目 Vite 栈天然配套，后续 T1/T2 前端测试也可复用）。交付时在 README/PR 说明此改动。

---

## 合并与联调流程

```
1. 各 worktree 分支 push 后，在 main 合并：
   合并顺序：feature/codegen-diengine → feature/backend-service → feature/frontend-panel
2. 合并后先 curl 验后端（用 T2 交付的测试脚本，含 dummy forward 自检用例）
3. 再接前端联调（T1 已对桩文件/mock 先行开发）
4. 全链路验证：拖拽 MLP → PPO + CartPole → 开始训练 → 曲线 → 停止 → 下载模型
5. 冲突优先保留 T2/T3 代码
6. 合并后删除 worktree：git worktree remove ../dlp-T1 等
```

## MVP 里程碑

- **M3-MVP 完成**：网页拖拽 MLP → PPO + CartPole 训练出曲线 → 停止 → 下载模型（全链路）
- SAC 放 v2.2+ / MVP 之后（需连续环境如 Pendulum，MVP 不做）

