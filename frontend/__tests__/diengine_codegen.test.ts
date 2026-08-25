import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { dirname, join } from "node:path";
import { fileURLToPath } from "node:url";
import { beforeAll, describe, expect, it } from "vitest";
import { generateDiEngineModel } from "../src/utils/codeCompileDiEngine";

const here = dirname(fileURLToPath(import.meta.url));
const fixturesDir = join(here, "fixtures");
const sampleGraphPath = join(fixturesDir, "sample_graph.json");
const sampleModelPath = join(fixturesDir, "sample_model.py");
const sampleShapesPath = join(fixturesDir, "sample_shapes.json");

function readSampleGraph(): { nodes: unknown[]; edges: unknown[] } {
  return JSON.parse(readFileSync(sampleGraphPath, "utf-8"));
}

describe("generateDiEngineModel", () => {
  let result: ReturnType<typeof generateDiEngineModel>;

  beforeAll(() => {
    const graph = readSampleGraph();
    result = generateDiEngineModel(graph.nodes as never, graph.edges as never);

    // Pipe the generator output into filesystem fixtures so the Python layer
    // can exec the produced code (node cannot run torch).
    mkdirSync(fixturesDir, { recursive: true });
    writeFileSync(sampleModelPath, result.model_code, "utf-8");
    writeFileSync(
      sampleShapesPath,
      `${JSON.stringify(
        { obs_shape: result.obs_shape, action_shape: result.action_shape },
        null,
        2
      )}\n`,
      "utf-8"
    );
  });

  it("derives obs_shape and action_shape from the graph", () => {
    expect(result.obs_shape).toBe(4);
    expect(result.action_shape).toBe(2);
  });

  it("declares obs_shape/action_shape constructor parameters", () => {
    expect(result.model_code).toContain(
      'def __init__(self, obs_shape, action_shape, action_space="discrete"):'
    );
    expect(result.model_code).toContain("self.obs_shape = obs_shape");
    expect(result.model_code).toContain("self.action_shape = action_shape");
    expect(result.model_code).toContain("self.action_space = action_space");
    expect(result.model_code).toContain("in_features=obs_shape");
    expect(result.model_code).toContain("out_features=action_shape");
  });

  it("registers itself as user_net for the T2 training service", () => {
    expect(result.model_code).toContain("from ding.utils import MODEL_REGISTRY");
    expect(result.model_code).toContain('@MODEL_REGISTRY.register("user_net")');
    expect(result.model_code).toContain(
      'mode = ["compute_actor", "compute_critic", "compute_actor_critic"]'
    );
  });

  it("implements the three forward modes", () => {
    expect(result.model_code).toContain("def compute_actor(");
    expect(result.model_code).toContain("def compute_critic(");
    expect(result.model_code).toContain("def compute_actor_critic(");
    expect(result.model_code).toContain("if mode == 'compute_actor':");
    expect(result.model_code).toContain("if mode == 'compute_critic':");
    expect(result.model_code).toContain("if mode == 'compute_actor_critic':");
  });

  it("returns logit and value dict keys", () => {
    expect(result.model_code).toContain("return {'logit': logit}");
    expect(result.model_code).toContain("return {'value': value}");
    expect(result.model_code).toContain("return {'logit': logit, 'value': value}");
  });

  it("squeezes the critic value to a 1-D (B,) tensor", () => {
    expect(result.model_code).toContain("self.critic_head(x).squeeze(-1)");
  });

  it("keeps the actor output as raw logits (no tanh)", () => {
    expect(result.model_code).toContain("self.actor_head = nn.Linear(");
    expect(result.model_code).not.toContain("nn.Tanh()");
  });

  it("writes the generated model_code to the python fixture", () => {
    expect(readFileSync(sampleModelPath, "utf-8")).toBe(result.model_code);
  });
});
