import type { Edge, Node } from "@xyflow/react";

/**
 * Generate a DI-engine compliant model class from the visual graph.
 *
 * Contract (see docs/taskcard): this is the SINGLE source of `model_code` for
 * training submissions. T1 must call this (never the legacy codeCompile.ts),
 * and T2 only exec-loads + registers the emitted class via MODEL_REGISTRY.
 *
 * The emitted class follows the discrete actor-critic interface of
 * `ding/model/template/vac.py`:
 *   - `forward(x, mode)` dispatches to three modes
 *   - `'compute_actor'`        -> {'logit': Tensor (B, action_shape)}   (raw logits, no tanh)
 *   - `'compute_critic'`       -> {'value': Tensor (B,)}                (squeezed)
 *   - `'compute_actor_critic'` -> {'logit': ..., 'value': ...}
 *
 * MVP scope: MLP only (input -> Linear/activation ... -> Linear).
 * `obs_shape` is derived from the input node, `action_shape` from the output node.
 */
export function generateDiEngineModel(
  nodes: Node[],
  edges: Edge[]
): { model_code: string; obs_shape: number; action_shape: number } {
  const topLevel = nodes.filter(n => !n.parentId);
  const byId = new Map(topLevel.map(n => [n.id, n]));

  const inDeg: Record<string, number> = {};
  const outDeg: Record<string, number> = {};
  topLevel.forEach(n => {
    inDeg[n.id] = 0;
    outDeg[n.id] = 0;
  });
  edges.forEach(e => {
    if (byId.has(e.source) && byId.has(e.target)) {
      outDeg[e.source] += 1;
      inDeg[e.target] += 1;
    }
  });

  const ordered = topologicalSort(topLevel, edges);
  const inputNodes = ordered.filter(n => inDeg[n.id] === 0);
  const outputNodes = ordered.filter(n => outDeg[n.id] === 0);

  const firstLinear = ordered.find(n => n.type === "linear_layer");

  let obsShape = deriveObsShape(inputNodes);
  if (obsShape <= 0 && firstLinear) {
    obsShape = toNumber(firstLinear.data?.in_features);
  }
  if (obsShape <= 0) obsShape = 1;

  const outputNode = outputNodes[0] ?? ordered[ordered.length - 1];
  let actionShape = deriveActionShape(outputNode);
  if (actionShape <= 0) actionShape = 1;

  const hidden = deriveHidden(outputNode, ordered, obsShape);

  const inputIds = new Set(inputNodes.map(n => n.id));
  const backbone = ordered.filter(n => !inputIds.has(n.id) && n.id !== outputNode?.id);

  const modelCode = renderModel(backbone, hidden);
  return { model_code: modelCode, obs_shape: obsShape, action_shape: actionShape };
}

function topologicalSort(nodes: Node[], edges: Edge[]): Node[] {
  const adj: Record<string, string[]> = {};
  const inDeg: Record<string, number> = {};
  nodes.forEach(n => {
    adj[n.id] = [];
    inDeg[n.id] = 0;
  });
  edges.forEach(e => {
    if (adj[e.source] && adj[e.target] !== undefined) {
      adj[e.source].push(e.target);
      inDeg[e.target] += 1;
    }
  });

  const queue = nodes.filter(n => inDeg[n.id] === 0).map(n => n.id);
  const sorted: string[] = [];
  while (queue.length > 0) {
    const u = queue.shift()!;
    sorted.push(u);
    adj[u].forEach(v => {
      inDeg[v] -= 1;
      if (inDeg[v] === 0) queue.push(v);
    });
  }

  const byId = new Map(nodes.map(n => [n.id, n]));
  const remaining = nodes.map(n => n.id).filter(id => !sorted.includes(id));
  return [...sorted, ...remaining]
    .map(id => byId.get(id))
    .filter((n): n is Node => n !== undefined);
}

function deriveObsShape(inputNodes: Node[]): number {
  for (const n of inputNodes) {
    const dims = n.data?.dims;
    if (!Array.isArray(dims) || dims.length === 0) continue;
    const feats = (dims as Array<{ size?: unknown; type?: string }>)
      .filter(d => d?.type !== "batch")
      .map(d => toNumber(d?.size))
      .filter(v => v > 0);
    if (feats.length > 0) return feats.reduce((a, b) => a * b, 1);
  }
  return 0;
}

function deriveActionShape(outputNode: Node | undefined): number {
  if (!outputNode) return 0;
  const out = toNumber(outputNode.data?.out_features);
  if (out > 0) return out;
  const shape = outputNode.data?.__shape;
  if (Array.isArray(shape) && shape.length > 0) {
    const last = toNumber(shape[shape.length - 1]);
    if (last > 0) return last;
  }
  return 0;
}

function deriveHidden(outputNode: Node | undefined, ordered: Node[], fallback: number): number {
  if (outputNode) {
    const inFeat = toNumber(outputNode.data?.in_features);
    if (inFeat > 0) return inFeat;
  }
  for (let i = ordered.length - 1; i >= 0; i--) {
    if (outputNode && ordered[i].id === outputNode.id) continue;
    const out = toNumber(ordered[i].data?.out_features);
    if (out > 0) return out;
  }
  return fallback;
}

const ACTIVATION_INIT: Record<string, string> = {
  relu_layer: "nn.ReLU()",
  leakyrelu_layer: "nn.LeakyReLU()",
  gelu_layer: "nn.GELU()",
  elu_layer: "nn.ELU()",
  selu_layer: "nn.SELU()",
  tanh_layer: "nn.Tanh()",
  sigmoid_layer: "nn.Sigmoid()",
  softplus_layer: "nn.Softplus()",
  softsign_layer: "nn.Softsign()",
  hardswish_layer: "nn.Hardswish()",
  hardsigmoid_layer: "nn.Hardsigmoid()",
};

function toNumber(v: unknown): number {
  const n = Number(v);
  return Number.isFinite(n) ? n : 0;
}

function renderModel(backbone: Node[], hidden: number): string {
  const lines: string[] = [];
  lines.push("import torch");
  lines.push("import torch.nn as nn");
  lines.push("");
  lines.push("from ding.utils import MODEL_REGISTRY");
  lines.push("");
  lines.push("");
  lines.push('@MODEL_REGISTRY.register("user_net")');
  lines.push("class DiEngineModel(nn.Module):");
  lines.push('    """');
  lines.push("    DI-engine compliant actor-critic model generated from the visual graph.");
  lines.push('    """');
  lines.push('    mode = ["compute_actor", "compute_critic", "compute_actor_critic"]');
  lines.push("");
  lines.push('    def __init__(self, obs_shape, action_shape, action_space="discrete"):');
  lines.push("        super().__init__()");
  lines.push("        self.obs_shape = obs_shape");
  lines.push("        self.action_shape = action_shape");
  lines.push("        self.action_space = action_space");
  lines.push("");

  let firstLinearEmitted = false;
  backbone.forEach((n, i) => {
    const isFirstLinear = n.type === "linear_layer" && !firstLinearEmitted;
    const init = renderLayerInit(n, i, isFirstLinear);
    if (n.type === "linear_layer") firstLinearEmitted = true;
    lines.push(init);
  });

  lines.push(`        self.actor_head = nn.Linear(in_features=${hidden}, out_features=action_shape)`);
  lines.push(`        self.critic_head = nn.Linear(in_features=${hidden}, out_features=1)`);
  lines.push("");
  lines.push("    def forward(self, x, mode):");
  lines.push("        if mode == 'compute_actor':");
  lines.push("            return self.compute_actor(x)");
  lines.push("        if mode == 'compute_critic':");
  lines.push("            return self.compute_critic(x)");
  lines.push("        if mode == 'compute_actor_critic':");
  lines.push("            return self.compute_actor_critic(x)");
  lines.push('        raise ValueError("unsupported forward mode: {}".format(mode))');
  lines.push("");
  lines.push("    def _encode(self, x):");
  if (backbone.length === 0) {
    lines.push("        return x");
  } else {
    backbone.forEach((_, i) => {
      lines.push(`        x = self.layer_${i}(x)`);
    });
    lines.push("        return x");
  }
  lines.push("");
  lines.push("    def compute_actor(self, x):");
  lines.push("        x = self._encode(x)");
  lines.push("        logit = self.actor_head(x)");
  lines.push("        return {'logit': logit}");
  lines.push("");
  lines.push("    def compute_critic(self, x):");
  lines.push("        x = self._encode(x)");
  lines.push("        value = self.critic_head(x).squeeze(-1)");
  lines.push("        return {'value': value}");
  lines.push("");
  lines.push("    def compute_actor_critic(self, x):");
  lines.push("        x = self._encode(x)");
  lines.push("        logit = self.actor_head(x)");
  lines.push("        value = self.critic_head(x).squeeze(-1)");
  lines.push("        return {'logit': logit, 'value': value}");
  lines.push("");

  return lines.join("\n");
}

function renderLayerInit(node: Node, index: number, isFirstLinear: boolean): string {
  const name = `layer_${index}`;
  const type = node.type;
  const data = node.data ?? {};

  if (type === "linear_layer") {
    const inFeat = isFirstLinear ? "obs_shape" : toNumber(data.in_features);
    const outFeat = toNumber(data.out_features);
    const bias = data.bias === false ? ", bias=False" : "";
    return `        self.${name} = nn.Linear(in_features=${inFeat}, out_features=${outFeat}${bias})`;
  }

  const activation = type ? ACTIVATION_INIT[type] : undefined;
  if (activation) {
    return `        self.${name} = ${activation}`;
  }

  throw new Error(
    `Unsupported layer type for DI-engine MVP (MLP only): ${type ?? "undefined"}`
  );
}
