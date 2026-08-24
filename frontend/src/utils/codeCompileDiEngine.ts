import type { Edge, Node } from "@xyflow/react";

/**
 * Generate a DI-engine compliant model class from the visual graph.
 *
 * Contract (see docs/taskcard): this is the SINGLE source of `model_code` for
 * training submissions. T1 must call this (never the legacy codeCompile.ts),
 * and T2 only exec-loads + registers the emitted class via MODEL_REGISTRY.
 *
 * Implemented by T3 — stub only, signature is frozen.
 */
export function generateDiEngineModel(
  nodes: Node[],
  edges: Edge[]
): { model_code: string; obs_shape: number; action_shape: number } {
  throw new Error("not implemented, see T3");
}
