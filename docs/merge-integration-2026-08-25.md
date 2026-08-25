# 合并与联调工作记录（2026-08-25）

> 分支：`main`（fork `hongfeiyao06-source/DL-Playground`）
> 结果：MVP 全链路跑通，打 tag `mvp-v1`（@ `f585341`）

---

## 一、合并

按任务卡顺序把三个 feature 分支合并进 `main`：

| 分支 | 冲突 | 处理 |
|------|------|------|
| `feature/codegen-diengine`（T3） | `codeCompileDiEngine.ts` | 取 T3 完整实现 |
| `feature/backend-service`（T2） | 无 | — |
| `feature/frontend-panel`（T1） | `codeCompileDiEngine.ts` | 保留 T3（T1 的 `void` 修复已被 T3 实现取代） |

## 二、联调验证

- 后端 22/22 验收测试：`.sh`（Cygwin）与 `.ps1`（PowerShell 5.1）各独立跑通。
- 浏览器全链路：拖拽 MLP → `generateDiEngineModel` → POST `/start` → 训练到 `done`（CartPole reward 200）→ curve 真实数据 → 下载 `model.pth`。

## 三、联调中发现并修复的 bug（8 处）

| # | 问题 | 修复 |
|---|------|------|
| 1 | 独立 `training_service:app` 缺 CORS，浏览器跨域被拦 | 补 `CORSMiddleware` |
| 2 | 曲线图无坐标轴、看不出数值 | 手写 SVG 加 X/Y 轴 + 刻度 + 轴标题 |
| 3 | 生成的 DI-engine 模型代码在 UI 不可见 | Training 面板加折叠展示区（`obs_shape`/`action_shape` + `model_code`） |
| 4 | 生成代码不能 copy/download | 补 Copy（`navigator.clipboard`）/ Download（Blob）按钮 |
| 5 | Training 面板无法折叠 | 修 onClick（原绑定到 `hyperparamsOpen`）+ 箭头状态 |
| 6 | 空图 / 无 Linear 图静默生成退化模型 | 生成器加图校验，抛清晰错误 |
| 7 | 侧边栏拖拽条：拖拽选中文字 + 松手后还跟鼠标 | 补 `preventDefault()` + `userSelect:none` |
| 8 | CodePanel 拖拽条同类隐患 | 补 `userSelect:none` |

## 四、收尾

- 删除 3 个 worktree（`dlp-T1/T2/T3`）+ 本地 feature 分支（fork 上保留）。
- 打 tag `mvp-v1` 并 push。
- 停后台服务（后端 8000 / 前端 5173）。

## 五、最终状态

```
fork: hongfeiyao06-source/DL-Playground
  main                    f585341   ← MVP 全部代码
  tag mvp-v1              f585341   ← MVP 基线
  feature/backend-service 99ffa68   ← fork 保留
  feature/codegen-diengine 065071a  ← fork 保留
  feature/frontend-panel  6cb5256   ← fork 保留
```

## 六、遗留（非阻塞，属 MVP 范围外 / 体验项）

1. 曲线图训练早期单点阶段，X 轴刻度可能显示「0.5」（迭代 0.5 无意义，多点后正常）。
2. 前端 `BASE_URL` 硬编码 `http://localhost:8000`，部署需改环境变量。
3. 前端未校验 `obs_shape` 是否匹配 CartPole（=4），设错会训练报错。
4. 未来功能（SAC / Pendulum / Conv / LSTM 等）从 `main` 拉新分支 + worktree 开发。
