# T0 交接说明（给 T1 / T2 / T3 代理）

> 本文记录 T0 收尾后仓库的**实际状态**，以及后续代理开工前必须知道的、和任务卡原文不一致的几件事。
> 契约全文见 [`docs/taskcard-v2.5.md`](./taskcard-v2.5.md)。

---

## 1. 仓库 / 分支 / remote 现状

| 项 | 值 |
|----|----|
| 仓库根 | `D:/WorkSpace/NN_Play`（**已从 `D:/WorkSpace/DL-Playground` 迁来**，旧目录已删除） |
| worktree | `D:/WorkSpace/dlp-T1` → `feature/frontend-panel`<br>`D:/WorkSpace/dlp-T2` → `feature/backend-service`<br>`D:/WorkSpace/dlp-T3` → `feature/codegen-diengine` |
| origin | `git@github.com:hongfeiyao06-source/DL-Playground.git`（团队 fork） |
| upstream | `git@github.com:dsgiitr/DL-Playground.git`（上游） |
| push 状态 | `main`（63d9993）+ 三个 feature 分支（274450b）均已 push 到 origin |

## 2. ⚠️ 契约文档不在 feature 分支上

feature 分支停在 `274450b`，**没有** `docs/taskcard-v2.5.md`（它只在 main 上）。在任意 worktree 里这样读契约全文：

```bash
git show origin/main:docs/taskcard-v2.5.md
git show origin/main:docs/handoff-T0.md      # 本文
```

或直接读桌面原文件 `C:\Users\Administrator\Desktop\DL-Playground-DIengine融合MVP-任务卡.md`。

## 3. 环境事实（开工第一步先 `conda env list` 确认）

本机有**两个 Anaconda**，别搞混：

| 环境 | 路径 | 用途 |
|------|------|------|
| `ding_env` (py3.10) | `D:\anaconda3\envs\ding_env` | DI-engine 所在地（T2/T3 用） |
| `dlbackend` (py3.11) | `D:\anaconda3\envs\dlbackend` | 后端 FastAPI 进程（不 import ding） |

- **ding python 路径不要硬编码**，用环境变量 `DING_PYTHON`，默认 `D:\anaconda3\envs\ding_env\python.exe`。
- 前端 `npm run dev`（Vite 默认 `http://localhost:5173`）；后端 `conda activate dlbackend && uvicorn runner:app --host 0.0.0.0 --port 8000`。

## 4. 已就位、无需代理操心的

- 桩文件 `frontend/src/utils/codeCompileDiEngine.ts` 已在**全部分支**：T1 直接 import（编译期必须通过）；T3 直接填实现，**签名冻结**（`generateDiEngineModel(nodes, edges): { model_code; obs_shape; action_shape }`）。
- 三个 feature 分支已 push，代理可直接从 fork 拉。

## 5. T0 期间做的决策（与任务卡字面不同处）

1. 任务卡提交为 **`docs/taskcard-v2.5.md`**（步骤 7 原文写的是 v2.4，但文件实际是 v2.5 最终定稿）。
2. 桩文件 import 风格对齐现有 `codeCompile.ts`：`import type { Edge, Node } from "@xyflow/react";`。
3. README 已补「本地开发启动」章节（前端 / 后端 / Docker 构建 / DING_PYTHON）。

## 6. 合并阶段提醒（照任务卡）

合并顺序：`feature/codegen-diengine` → `feature/backend-service` → `feature/frontend-panel`；冲突优先保留 T2/T3 代码。
