import type { ChangeEvent } from "react";
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

function buildPolyline(
    xs: number[],
    ys: number[],
    width: number,
    height: number,
    padding: number
): string {
    if (xs.length === 0 || ys.length === 0) return "";
    const minX = xs[0];
    const maxX = xs[xs.length - 1];
    const minY = Math.min(...ys);
    const maxY = Math.max(...ys);
    const xSpan = maxX === minX ? 1 : maxX - minX;
    const ySpan = maxY === minY ? 1 : maxY - minY;

    const points = xs.map((x, i) => {
        const nx = padding + ((x - minX) / xSpan) * (width - 2 * padding);
        const ny = padding + (1 - (ys[i] - minY) / ySpan) * (height - 2 * padding);
        return `${nx},${ny}`;
    });
    return points.join(" ");
}

function SvgCurve({ curve }: { curve: TrainingCurve }) {
    const width = 260;
    const height = 140;
    const padding = 24;
    const hasData = curve.iterations.length > 0 && curve.eval_rewards.length > 0;

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
                        <polyline
                            fill="none"
                            stroke={THEME.accent}
                            strokeWidth={2}
                            points={buildPolyline(curve.iterations, curve.eval_rewards, width, height, padding)}
                        />
                        <circle
                            r={3}
                            fill={THEME.accent}
                            cx={padding}
                            cy={padding + (1 - (curve.eval_rewards[0] - Math.min(...curve.eval_rewards)) / (Math.max(...curve.eval_rewards) - Math.min(...curve.eval_rewards))) * (height - 2 * padding)}
                        />
                    </>
                ) : (
                    <text
                        x={width / 2}
                        y={height / 2}
                        textAnchor="middle"
                        fill={THEME.textSecondary}
                        fontSize="12"
                    >
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
    } = props;

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
                onClick={() => setHyperparamsOpen(!hyperparamsOpen)}
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
                    marginBottom: "10px",
                }}
                onMouseEnter={e => e.currentTarget.style.color = THEME.textPrimary}
                onMouseLeave={e => e.currentTarget.style.color = THEME.textSecondary}
            >
                <ChevronIcon open={true} />
                Training
            </div>

            <div style={{
                display: "flex",
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
