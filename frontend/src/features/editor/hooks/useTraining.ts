import { useCallback, useEffect, useRef, useState } from "react";
import type { Edge, Node } from "@xyflow/react";
import { generateDiEngineModel } from "../../../utils/codeCompileDiEngine";

const BASE_URL = "http://localhost:8000";
const POLL_INTERVAL_MS = 2000;

export type TrainingStatus = "starting" | "running" | "done" | "error";

export type TrainingCurve = {
    iterations: number[];
    eval_rewards: number[];
    losses: number[];
};

export type TrainingState = {
    algorithm: string;
    env: string;
    hyperparamsOpen: boolean;
    learningRate: number;
    batchSize: number;
    maxTrainIter: number;
    taskId: string | null;
    status: TrainingStatus | null;
    trainIter: number;
    evalReward: number | null;
    message: string;
    curve: TrainingCurve;
    loading: boolean;
    backendAvailable: boolean;
    error: string | null;
};

const INITIAL_CURVE: TrainingCurve = {
    iterations: [],
    eval_rewards: [],
    losses: [],
};

export function useTraining(nodes: Node[], edges: Edge[]) {
    const [algorithm, setAlgorithm] = useState("ppo");
    const [env, setEnv] = useState("cartpole");
    const [hyperparamsOpen, setHyperparamsOpen] = useState(false);
    const [learningRate, setLearningRate] = useState(0.001);
    const [batchSize, setBatchSize] = useState(256);
    const [maxTrainIter, setMaxTrainIter] = useState(5000);

    const [taskId, setTaskId] = useState<string | null>(null);
    const [status, setStatus] = useState<TrainingStatus | null>(null);
    const [trainIter, setTrainIter] = useState(0);
    const [evalReward, setEvalReward] = useState<number | null>(null);
    const [message, setMessage] = useState("");
    const [curve, setCurve] = useState<TrainingCurve>(INITIAL_CURVE);
    const [loading, setLoading] = useState(false);
    const [backendAvailable, setBackendAvailable] = useState(true);
    const [error, setError] = useState<string | null>(null);
    const [generatedModelCode, setGeneratedModelCode] = useState<string | null>(null);
    const [generatedObsShape, setGeneratedObsShape] = useState<number | null>(null);
    const [generatedActionShape, setGeneratedActionShape] = useState<number | null>(null);

    const pollingRef = useRef<ReturnType<typeof setInterval> | null>(null);
    const abortRef = useRef<AbortController | null>(null);

    const clearPolling = useCallback(() => {
        if (pollingRef.current) {
            clearInterval(pollingRef.current);
            pollingRef.current = null;
        }
        if (abortRef.current) {
            abortRef.current.abort();
            abortRef.current = null;
        }
    }, []);

    const resetTraining = useCallback(() => {
        clearPolling();
        setTaskId(null);
        setStatus(null);
        setTrainIter(0);
        setEvalReward(null);
        setMessage("");
        setCurve(INITIAL_CURVE);
        setLoading(false);
        setError(null);
        setGeneratedModelCode(null);
        setGeneratedObsShape(null);
        setGeneratedActionShape(null);
    }, [clearPolling]);

    const fetchStatus = useCallback(async (id: string) => {
        try {
            const res = await fetch(`${BASE_URL}/api/training/${id}/status`, {
                signal: abortRef.current?.signal,
            });
            if (!res.ok) {
                if (res.status === 404) {
                    setStatus("error");
                    setMessage("Task not found on backend.");
                    clearPolling();
                }
                return;
            }
            const data = (await res.json()) as {
                status: TrainingStatus;
                train_iter: number;
                eval_reward: number | null;
                message: string;
            };
            setStatus(data.status);
            setTrainIter(data.train_iter ?? 0);
            setEvalReward(data.eval_reward ?? null);
            setMessage(data.message ?? "");
            if (data.status === "done" || data.status === "error") {
                clearPolling();
            }
        } catch (err) {
            if (err instanceof DOMException && err.name === "AbortError") return;
            setBackendAvailable(false);
            setStatus("error");
            setMessage("Backend unreachable during status poll.");
            clearPolling();
        }
    }, [clearPolling]);

    const fetchCurve = useCallback(async (id: string) => {
        try {
            const res = await fetch(`${BASE_URL}/api/training/${id}/curve`, {
                signal: abortRef.current?.signal,
            });
            if (!res.ok) return;
            const data = (await res.json()) as TrainingCurve;
            setCurve({
                iterations: Array.isArray(data.iterations) ? data.iterations : [],
                eval_rewards: Array.isArray(data.eval_rewards) ? data.eval_rewards : [],
                losses: Array.isArray(data.losses) ? data.losses : [],
            });
        } catch (err) {
            if (err instanceof DOMException && err.name === "AbortError") return;
            // Curve fetch failures are non-fatal; keep last known curve.
        }
    }, []);

    const startPolling = useCallback((id: string) => {
        clearPolling();
        abortRef.current = new AbortController();
        void fetchStatus(id);
        void fetchCurve(id);
        pollingRef.current = setInterval(() => {
            void fetchStatus(id);
            void fetchCurve(id);
        }, POLL_INTERVAL_MS);
    }, [clearPolling, fetchStatus, fetchCurve]);

    const startTraining = useCallback(async () => {
        resetTraining();
        setLoading(true);
        setBackendAvailable(true);

        try {
            let modelCode: string;
            let obsShape: number;
            let actionShape: number;
            try {
                const generated = generateDiEngineModel(nodes, edges);
                modelCode = generated.model_code;
                obsShape = generated.obs_shape;
                actionShape = generated.action_shape;
                setGeneratedModelCode(generated.model_code);
                setGeneratedObsShape(generated.obs_shape);
                setGeneratedActionShape(generated.action_shape);
            } catch (genErr) {
                // Stub throws at runtime; surface a clear message without crashing.
                const msg = genErr instanceof Error ? genErr.message : String(genErr);
                setStatus("error");
                setError(`Model generator not ready: ${msg}`);
                setLoading(false);
                return;
            }

            const res = await fetch(`${BASE_URL}/api/training/start`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({
                    network: {
                        model_code: modelCode,
                        obs_shape: obsShape,
                        action_shape: actionShape,
                    },
                    algorithm,
                    env,
                    hyperparams: {
                        learning_rate: learningRate,
                        batch_size: batchSize,
                        max_train_iter: maxTrainIter,
                    },
                }),
            });

            if (!res.ok) {
                const text = await res.text();
                setStatus("error");
                setError(text || `Start failed: ${res.status} ${res.statusText}`);
                setLoading(false);
                return;
            }

            const data = (await res.json()) as { task_id: string };
            setTaskId(data.task_id);
            setStatus("starting");
            setLoading(false);
            startPolling(data.task_id);
        } catch (err) {
            setBackendAvailable(false);
            setStatus("error");
            setError(err instanceof Error ? err.message : "Backend unavailable");
            setLoading(false);
        }
    }, [
        nodes, edges, algorithm, env, learningRate, batchSize, maxTrainIter,
        resetTraining, startPolling,
    ]);

    const stopTraining = useCallback(async () => {
        if (!taskId) return;
        try {
            const res = await fetch(`${BASE_URL}/api/training/${taskId}/stop`, {
                method: "POST",
            });
            if (!res.ok) {
                const text = await res.text();
                setError(text || `Stop failed: ${res.status} ${res.statusText}`);
                return;
            }
            await fetchStatus(taskId);
            clearPolling();
        } catch (err) {
            setBackendAvailable(false);
            setError(err instanceof Error ? err.message : "Backend unavailable");
        }
    }, [taskId, fetchStatus, clearPolling]);

    useEffect(() => {
        return () => {
            clearPolling();
        };
    }, [clearPolling]);

    const canStart = !loading && status !== "running" && status !== "starting";
    const canStop = status === "running" || status === "starting";
    const isDone = status === "done";

    return {
        algorithm,
        setAlgorithm,
        env,
        setEnv,
        hyperparamsOpen,
        setHyperparamsOpen,
        learningRate,
        setLearningRate,
        batchSize,
        setBatchSize,
        maxTrainIter,
        setMaxTrainIter,
        taskId,
        status,
        trainIter,
        evalReward,
        message,
        curve,
        loading,
        backendAvailable,
        error,
        startTraining,
        stopTraining,
        canStart,
        canStop,
        isDone,
        generatedModelCode,
        generatedObsShape,
        generatedActionShape,
        modelDownloadUrl: taskId ? `${BASE_URL}/api/training/${taskId}/model` : null,
    };
}
