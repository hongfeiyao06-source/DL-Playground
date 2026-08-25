import { useState, type ChangeEvent } from "react";
import type { TrainingCurve, TrainingStatus } from "../hooks/useTraining";

const THEME = {
    bg: "#18181bff",
    bgSection: "#27272a",
    border: "#64646dff",
    textPrimary: "#fdfdfdde",
    textSecondary: "#ccccd3ff",
    accent: "#0ea5e9",
    accentHover: "#0284c7",
    danger: "#ef4444",
    dangerHover: "#dc2626",
    success: "#22c55e",
    hover: "#3f3f46",
    itemBg: "#27272a",
    disabled: "#52525b",
};

const ChevronIcon = ({ open }: { open: boolean }) => (
    <svg
        width="16" height="16" viewBox="0 0 24 24"
        fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round"
        style={{
            transform: open ? "rotate(90deg)" : "rotate(0deg)",
            transition: "transform 0.2s ease",
            opacity: 0.7,
        }}
    >
        <polyline points="9 18 15 12 9 6"></polyline>
    </svg>
);

type AlgorithmOption = { value: string; label: string; disabled?: boolean; hint?: string };
type EnvOption = { value: string; label: string; disabled?: boolean; hint?: string };

const ALGORITHMS: AlgorithmOption[] = [
    { value: "ppo", label: "PPO" },
    { value: "sac", label: "SAC", disabled: true, hint: "即将支持" },
    { value: "iql", label: "IQL", disabled: true, hint: "即将支持" },
    { value: "maddpg", label: "MADDPG", disabled: true, hint: "即将支持" },
];

const ENVS: EnvOption[] = [
    { value: "cartpole", label: "CartPole" },
    { value: "pendulum", label: "Pendulum", disabled: true, hint: "即将支持" },
    { value: "hopper", label: "Hopper", disabled: true, hint: "即将支持" },
];

function statusLabel(status: TrainingStatus | null): string {
    if (!status) return "Idle";
    if (status === "starting") return "Starting…";
    if (status === "running") return "Training…";
    if (status === "done") return "Done";
    return "Error";
}

function niceStep(span: number, targetTicks: number): number {
    if (span <= 0) return 1;
    const raw = span / targetTicks;
    const mag = Math.pow(10, Math.floor(Math.log10(raw)));
    const norm = raw / mag;
    const nice = norm >= 5 ? 10 : norm >= 2 ? 5 : norm >= 1 ? 2 : 1;
    return nice * mag;
}

function makeTicks(min: number, max: number, targetTicks: number): number[] {
    const step = niceStep(max - min, targetTicks);
    const ticks: number[] = [];
    for (let v = Math.ceil(min / step) * step; v <= max + 1e-9; v += step) {
        ticks.push(Number(v.toFixed(4)));
    }
    if (ticks.length === 0) ticks.push(min);
    return ticks;
}

function formatTick(t: number): string {
    return Number.isInteger(t) ? String(t) : t.toFixed(1);
}

function SvgCurve({ curve }: { curve: TrainingCurve }) {
    const width = 280;
    const height = 180;
    const padL = 38;
    const padR = 10;
    const padT = 10;
    const padB = 30;

    const xs = curve.iterations;
    const ys = curve.eval_rewards;
    const hasData = xs.length > 0 && ys.length > 0;

    const xMin = 0;
    const xMax = Math.max(...xs, 1);
    const yMin = 0;
    const yMax = Math.max(...ys, 1);

    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const sx = (x: number) => padL + ((x - xMin) / (xMax - xMin)) * plotW;
    const sy = (y: number) => padT + (1 - (y - yMin) / (yMax - yMin)) * plotH;

    const xTicks = makeTicks(xMin, xMax, 5);
    const yTicks = makeTicks(yMin, yMax, 5);
    const points = xs.map((x, i) => `${sx(x)},${sy(ys[i])}`).join(" ");

    return (
        <div style={{
            background: THEME.bg,
            border: `1px solid ${THEME.border}`,
            borderRadius: "6px",
            padding: "10px",
            marginTop: "10px",
        }}>
            <div style={{ fontSize: "11px", color: THEME.textSecondary, marginBottom: "6px", fontWeight: 600 }}>
                Eval Reward Curve
            </div>
            <svg width={width} height={height} style={{ display: "block" }}>
                {hasData ? (
                    <>
                        <line x1={padL} y1={sy(yMin)} x2={width - padR} y2={sy(yMin)} stroke={THEME.border} strokeWidth={1} />
                        <line x1={padL} y1={padT} x2={padL} y2={sy(yMin)} stroke={THEME.border} strokeWidth={1} />
                        {xTicks.map((t, i) => (
                            <g key={`xt${i}`}>
                                <line x1={sx(t)} y1={sy(yMin)} x2={sx(t)} y2={sy(yMin) + 4} stroke={THEME.border} strokeWidth={1} />
                                <text x={sx(t)} y={sy(yMin) + 16} textAnchor="middle" fontSize="9" fill={THEME.textSecondary}>
                                    {formatTick(t)}
                                </text>
                            </g>
                        ))}
                        {yTicks.map((t, i) => (
                            <g key={`yt${i}`}>
                                <line x1={padL - 4} y1={sy(t)} x2={padL} y2={sy(t)} stroke={THEME.border} strokeWidth={1} />
                                <text x={padL - 8} y={sy(t) + 3} textAnchor="end" fontSize="9" fill={THEME.textSecondary}>
                                    {formatTick(t)}
                                </text>
                            </g>
                        ))}
                        <text x={padL + plotW / 2} y={height - 6} textAnchor="middle" fontSize="9" fill={THEME.textSecondary}>
                            iterations
                        </text>
                        <text x={14} y={padT + plotH / 2} textAnchor="middle" fontSize="9" fill={THEME.textSecondary} transform={`rotate(-90 14 ${padT + plotH / 2})`}>
                            eval_reward
                        </text>
                        <polyline fill="none" stroke={THEME.accent} strokeWidth={2} points={points} />
                        <circle r={3} fill={THEME.accent} cx={sx(xs[0])} cy={sy(ys[0])} />
                    </>
                ) : (
                    <text x={width / 2} y={height / 2} textAnchor="middle" fill={THEME.textSecondary} fontSize="12">
                        暂无数据
                    </text>
                )}
            </svg>
        </div>
    );
}

export type TrainingPanelProps = {
    algorithm: string;
    setAlgorithm: (v: string) => void;
    env: string;
    setEnv: (v: string) => void;
    hyperparamsOpen: boolean;
    setHyperparamsOpen: (v: boolean) => void;
    learningRate: number;
    setLearningRate: (v: number) => void;
    batchSize: number;
    setBatchSize: (v: number) => void;
    maxTrainIter: number;
    setMaxTrainIter: (v: number) => void;
    status: TrainingStatus | null;
    trainIter: number;
    evalReward: number | null;
    message: string;
    curve: TrainingCurve;
    loading: boolean;
    backendAvailable: boolean;
    error: string | null;
    startTraining: () => void;
    stopTraining: () => void;
    canStart: boolean;
    canStop: boolean;
    isDone: boolean;
    modelDownloadUrl: string | null;
    generatedModelCode: string | null;
    generatedObsShape: number | null;
    generatedActionShape: number | null;
};

export function TrainingPanel(props: TrainingPanelProps) {
    const {
        algorithm, setAlgorithm,
        env, setEnv,
        hyperparamsOpen, setHyperparamsOpen,
        learningRate, setLearningRate,
        batchSize, setBatchSize,
        maxTrainIter, setMaxTrainIter,
        status, trainIter, evalReward, message, curve,
        loading, backendAvailable, error,
        startTraining, stopTraining,
        canStart, canStop, isDone,
        modelDownloadUrl,
        generatedModelCode, generatedObsShape, generatedActionShape,
    } = props;

    const [trainingOpen, setTrainingOpen] = useState(true);
    const [generatedCodeOpen, setGeneratedCodeOpen] = useState(false);

    const onDownloadGeneratedCode = () => {
        if (!generatedModelCode) return;
        const blob = new Blob([generatedModelCode], { type: "text/x-python" });
        const url = URL.createObjectURL(blob);
        const a = document.createElement("a");
        a.href = url;
        a.download = "diengine_model.py";
        a.click();
        URL.revokeObjectURL(url);
    };

    const onAlgorithmChange = (e: ChangeEvent<HTMLSelectElement>) => {
        const option = ALGORITHMS.find(o => o.value === e.target.value);
        if (option?.disabled) return;
        setAlgorithm(e.target.value);
    };

    const onEnvChange = (e: ChangeEvent<HTMLSelectElement>) => {
        const option = ENVS.find(o => o.value === e.target.value);
        if (option?.disabled) return;
        setEnv(e.target.value);
    };

    const statusColor =
        status === "error" ? THEME.danger :
        status === "done" ? THEME.success :
        status === "running" || status === "starting" ? THEME.accent :
        THEME.textSecondary;

    return (
        <div style={{ marginTop: "20px" }}>
            <div
                onClick={() => setTrainingOpen(!trainingOpen)}
                style={{
                    display: "flex",
                    alignItems: "center",
                    gap: "8px",
                    cursor: "pointer",
                    color: THEME.textSecondary,
                    fontSize: "12px",
                    fontWeight: 700,
                    textTransform: "uppercase",
                    letterSpacing: "0.5px",
                    marginBottom: trainingOpen ? "10px" : "0px",
                }}
                onMouseEnter={e => e.currentTarget.style.color = THEME.textPrimary}
                onMouseLeave={e => e.currentTarget.style.color = THEME.textSecondary}
            >
                <ChevronIcon open={trainingOpen} />
                Training
            </div>

            <div style={{
                display: trainingOpen ? "flex" : "none",
                flexDirection: "column",
                gap: "10px",
                paddingLeft: "8px",
            }}>
                {/* Algorithm */}
                <label style={{ fontSize: "12px", color: THEME.textSecondary }}>
                    Algorithm
                    <select
                        value={algorithm}
                        onChange={onAlgorithmChange}
                        style={{
                            width: "100%",
                            marginTop: "4px",
                            padding: "8px",
                            background: THEME.itemBg,
                            color: THEME.textPrimary,
                            border: `1px solid ${THEME.border}`,
                            borderRadius: "6px",
                            fontSize: "13px",
                            outline: "none",
                        }}
                    >
                        {ALGORITHMS.map(opt => (
                            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                                {opt.label}{opt.hint ? ` (${opt.hint})` : ""}
                            </option>
                        ))}
                    </select>
                </label>

                {/* Environment */}
                <label style={{ fontSize: "12px", color: THEME.textSecondary }}>
                    Environment
                    <select
                        value={env}
                        onChange={onEnvChange}
                        style={{
                            width: "100%",
                            marginTop: "4px",
                            padding: "8px",
                            background: THEME.itemBg,
                            color: THEME.textPrimary,
                            border: `1px solid ${THEME.border}`,
                            borderRadius: "6px",
                            fontSize: "13px",
                            outline: "none",
                        }}
                    >
                        {ENVS.map(opt => (
                            <option key={opt.value} value={opt.value} disabled={opt.disabled}>
                                {opt.label}{opt.hint ? ` (${opt.hint})` : ""}
                            </option>
                        ))}
                    </select>
                </label>

                {/* Hyperparameters */}
                <div style={{
                    background: THEME.bg,
                    border: `1px solid ${THEME.border}`,
                    borderRadius: "6px",
                    overflow: "hidden",
                }}>
                    <button
                        onClick={() => setHyperparamsOpen(!hyperparamsOpen)}
                        style={{
                            width: "100%",
                            padding: "8px 10px",
                            background: "transparent",
                            border: "none",
                            color: THEME.textSecondary,
                            fontSize: "12px",
                            fontWeight: 600,
                            textAlign: "left",
                            cursor: "pointer",
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                        }}
                    >
                        Hyperparameters
                        <ChevronIcon open={hyperparamsOpen} />
                    </button>
                    {hyperparamsOpen && (
                        <div style={{ padding: "10px", display: "flex", flexDirection: "column", gap: "10px" }}>
                            <label style={{ fontSize: "12px", color: THEME.textSecondary }}>
                                Learning Rate
                                <input
                                    type="number"
                                    step={0.0001}
                                    value={learningRate}
                                    onChange={e => setLearningRate(Number(e.target.value))}
                                    style={{
                                        width: "100%",
                                        marginTop: "4px",
                                        padding: "8px",
                                        background: THEME.itemBg,
                                        color: THEME.textPrimary,
                                        border: `1px solid ${THEME.border}`,
                                        borderRadius: "6px",
                                        fontSize: "13px",
                                        outline: "none",
                                    }}
                                />
                            </label>
                            <label style={{ fontSize: "12px", color: THEME.textSecondary }}>
                                Batch Size
                                <input
                                    type="number"
                                    step={1}
                                    value={batchSize}
                                    onChange={e => setBatchSize(Number(e.target.value))}
                                    style={{
                                        width: "100%",
                                        marginTop: "4px",
                                        padding: "8px",
                                        background: THEME.itemBg,
                                        color: THEME.textPrimary,
                                        border: `1px solid ${THEME.border}`,
                                        borderRadius: "6px",
                                        fontSize: "13px",
                                        outline: "none",
                                    }}
                                />
                            </label>
                            <label style={{ fontSize: "12px", color: THEME.textSecondary }}>
                                Max Train Iter
                                <input
                                    type="number"
                                    step={100}
                                    value={maxTrainIter}
                                    onChange={e => setMaxTrainIter(Number(e.target.value))}
                                    style={{
                                        width: "100%",
                                        marginTop: "4px",
                                        padding: "8px",
                                        background: THEME.itemBg,
                                        color: THEME.textPrimary,
                                        border: `1px solid ${THEME.border}`,
                                        borderRadius: "6px",
                                        fontSize: "13px",
                                        outline: "none",
                                    }}
                                />
                            </label>
                        </div>
                    )}
                </div>

                {/* Actions */}
                <div style={{ display: "flex", gap: "8px" }}>
                    <button
                        onClick={startTraining}
                        disabled={!canStart}
                        style={{
                            flex: 1,
                            padding: "10px",
                            background: canStart ? THEME.accent : THEME.disabled,
                            color: "#fff",
                            border: "none",
                            borderRadius: "6px",
                            cursor: canStart ? "pointer" : "not-allowed",
                            fontSize: "13px",
                            fontWeight: 600,
                            transition: "background 0.2s",
                        }}
                        onMouseEnter={e => { if (canStart) e.currentTarget.style.background = THEME.accentHover; }}
                        onMouseLeave={e => { if (canStart) e.currentTarget.style.background = THEME.accent; }}
                    >
                        {loading ? "Starting…" : "Start Training"}
                    </button>
                    <button
                        onClick={stopTraining}
                        disabled={!canStop}
                        style={{
                            flex: 1,
                            padding: "10px",
                            background: canStop ? THEME.danger : THEME.disabled,
                            color: "#fff",
                            border: "none",
                            borderRadius: "6px",
                            cursor: canStop ? "pointer" : "not-allowed",
                            fontSize: "13px",
                            fontWeight: 600,
                            transition: "background 0.2s",
                        }}
                        onMouseEnter={e => { if (canStop) e.currentTarget.style.background = THEME.dangerHover; }}
                        onMouseLeave={e => { if (canStop) e.currentTarget.style.background = THEME.danger; }}
                    >
                        Stop
                    </button>
                </div>

                {/* Status */}
                {(status || message || error) && (
                    <div style={{
                        padding: "10px",
                        background: THEME.bg,
                        border: `1px solid ${THEME.border}`,
                        borderRadius: "6px",
                        fontSize: "12px",
                        color: THEME.textSecondary,
                    }}>
                        {status && (
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                                <span>Status</span>
                                <span style={{ color: statusColor, fontWeight: 600 }}>{statusLabel(status)}</span>
                            </div>
                        )}
                        {status && (status === "running" || status === "starting" || status === "done") && (
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                                <span>Iteration</span>
                                <span style={{ color: THEME.textPrimary, fontWeight: 500 }}>{trainIter}</span>
                            </div>
                        )}
                        {evalReward !== null && (
                            <div style={{ display: "flex", justifyContent: "space-between", marginBottom: "4px" }}>
                                <span>Eval Reward</span>
                                <span style={{ color: THEME.textPrimary, fontWeight: 500 }}>{evalReward.toFixed(2)}</span>
                            </div>
                        )}
                        {message && (
                            <div style={{ marginTop: "4px", color: THEME.textSecondary }}>{message}</div>
                        )}
                        {error && (
                            <div style={{ marginTop: "4px", color: THEME.danger }}>{error}</div>
                        )}
                    </div>
                )}

                {!backendAvailable && (
                    <div style={{
                        padding: "10px",
                        background: "rgba(239, 68, 68, 0.1)",
                        border: `1px solid ${THEME.danger}`,
                        borderRadius: "6px",
                        fontSize: "12px",
                        color: THEME.danger,
                    }}>
                        后端不可用 (Backend unavailable). Please start the backend service at http://localhost:8000.
                    </div>
                )}

                {/* Curve */}
                <SvgCurve curve={curve} />

                {/* Generated DI-engine model code */}
                {generatedModelCode && (
                    <div style={{
                        background: THEME.bg,
                        border: `1px solid ${THEME.border}`,
                        borderRadius: "6px",
                        overflow: "hidden",
                    }}>
                        <div style={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "space-between",
                            padding: "8px 10px",
                        }}>
                            <button
                                onClick={() => setGeneratedCodeOpen(!generatedCodeOpen)}
                                style={{
                                    background: "transparent",
                                    border: "none",
                                    color: THEME.textSecondary,
                                    fontSize: "12px",
                                    fontWeight: 600,
                                    cursor: "pointer",
                                    display: "flex",
                                    alignItems: "center",
                                    gap: "8px",
                                    padding: 0,
                                }}
                            >
                                <ChevronIcon open={generatedCodeOpen} />
                                生成的 DI-engine 模型代码
                            </button>
                            <div style={{ display: "flex", gap: "6px" }}>
                                <button
                                    onClick={() => navigator.clipboard.writeText(generatedModelCode)}
                                    style={{
                                        padding: "4px 8px",
                                        fontSize: "11px",
                                        cursor: "pointer",
                                        background: THEME.itemBg,
                                        color: THEME.textPrimary,
                                        border: `1px solid ${THEME.border}`,
                                        borderRadius: "4px",
                                    }}
                                >
                                    Copy
                                </button>
                                <button
                                    onClick={onDownloadGeneratedCode}
                                    style={{
                                        padding: "4px 8px",
                                        fontSize: "11px",
                                        cursor: "pointer",
                                        background: THEME.itemBg,
                                        color: THEME.textPrimary,
                                        border: `1px solid ${THEME.border}`,
                                        borderRadius: "4px",
                                    }}
                                >
                                    Download
                                </button>
                            </div>
                        </div>
                        {generatedCodeOpen && (
                            <div style={{ padding: "10px" }}>
                                <div style={{ fontSize: "11px", color: THEME.textSecondary, marginBottom: "6px" }}>
                                    obs_shape={generatedObsShape} · action_shape={generatedActionShape}
                                </div>
                                <pre style={{
                                    margin: 0,
                                    padding: "8px",
                                    background: "#111",
                                    border: `1px solid ${THEME.border}`,
                                    borderRadius: "4px",
                                    color: THEME.textPrimary,
                                    fontSize: "10px",
                                    lineHeight: 1.4,
                                    overflowX: "auto",
                                    maxHeight: "240px",
                                    overflowY: "auto",
                                    whiteSpace: "pre",
                                }}>
                                    {generatedModelCode}
                                </pre>
                            </div>
                        )}
                    </div>
                )}

                {/* Download */}
                {isDone && modelDownloadUrl && (
                    <a
                        href={modelDownloadUrl}
                        download
                        style={{
                            display: "block",
                            padding: "10px",
                            background: THEME.success,
                            color: "#fff",
                            textAlign: "center",
                            borderRadius: "6px",
                            fontSize: "13px",
                            fontWeight: 600,
                            textDecoration: "none",
                            transition: "background 0.2s",
                        }}
                        onMouseEnter={e => e.currentTarget.style.background = "#16a34a"}
                        onMouseLeave={e => e.currentTarget.style.background = THEME.success}
                    >
                        Download Model
                    </a>
                )}
            </div>
        </div>
    );
}
