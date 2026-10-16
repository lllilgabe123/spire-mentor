import { describe, expect, it, vi } from "vitest";
import { buildCommands, type CommandContext } from "./commands";

function ctx(over: Partial<CommandContext> = {}): CommandContext {
  const noop = () => {};
  return {
    mode: "play",
    hasLive: true,
    liveDecision: "combat_play",
    stepDecision: null,
    replayCount: 0,
    advisors: true,
    stateImport: true,
    staticMode: false,
    analyzeInTopbar: true,
    resume: null,
    bridgeConnected: false,
    bridgeDirect: false,
    viewedFloor: null,
    busy: {},
    actions: {
      save: noop, load: noop, importState: noop, importLive: noop, endRun: noop,
      analyze: noop, regretReview: noop, runReview: noop, searchTree: noop,
      simlabFromHere: noop, recommend: noop, playFromHere: noop,
      bridgeTree: noop, liveSearch: noop, bridgeSeek: noop,
    },
    ...over,
  };
}

const byId = (c: CommandContext) => Object.fromEntries(buildCommands(c).map((x) => [x.id, x]));

describe("buildCommands", () => {
  it("groups the run lifecycle under `run` with End run last and marked danger", () => {
    const run = buildCommands(ctx()).filter((c) => c.group === "run");
    expect(run.map((c) => c.id)).toEqual(["save", "load", "import-state", "end-run"]);
    expect(run[run.length - 1].danger).toBe(true);
  });

  it("offers Import from live game only while the game is connected", () => {
    expect(byId(ctx())["import-live"]).toBeUndefined();
    const on = buildCommands(ctx({ bridgeConnected: true })).filter((c) => c.group === "run");
    expect(on.map((c) => c.id)).toEqual(["save", "load", "import-state", "import-live", "end-run"]);
    expect(byId(ctx({ bridgeConnected: true, stateImport: false }))["import-live"]).toBeUndefined();
  });

  it("hides the play-only run commands in replay, and the replay-only ones in play", () => {
    const replay = byId(ctx({ mode: "replay", hasLive: false, replayCount: 10, stepDecision: "combat_play" }));
    expect(replay["save"]).toBeUndefined();
    expect(replay["end-run"]).toBeUndefined();
    expect(replay["recommend"]).toBeDefined();
    const play = byId(ctx());
    expect(play["recommend"]).toBeUndefined();
  });

  it("DISABLES (not hides) run-state-dependent analyses with a reason", () => {
    const cmds = byId(ctx({ liveDecision: "card_reward" }));
    expect(cmds["search-tree"]).toBeDefined();
    expect(cmds["search-tree"].disabledReason).toMatch(/combat/);
    expect(cmds["simlab-from-here"].disabledReason).toMatch(/combat/);
    const inCombat = byId(ctx());
    expect(inCombat["search-tree"].disabledReason).toBeUndefined();
  });

  it("disables Save and End run before a run has started", () => {
    const cmds = byId(ctx({ hasLive: false, liveDecision: null }));
    expect(cmds["save"].disabledReason).toMatch(/no run/);
    expect(cmds["end-run"].disabledReason).toMatch(/no run/);
    expect(cmds["load"].disabledReason).toBeUndefined();
  });

  it("drops every advisor command when advisors are off", () => {
    const ids = buildCommands(ctx({ advisors: false })).map((c) => c.id);
    expect(ids).not.toContain("regret-review");
    expect(ids).not.toContain("search-tree");
    expect(ids).toContain("end-run");
  });

  it("hides SimLab-from-here in the static build (no /api/live/position)", () => {
    expect(byId(ctx({ staticMode: true }))["simlab-from-here"]).toBeUndefined();
  });

  it("carries the server's reason when a recording cannot be resumed", () => {
    const cmds = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, stepDecision: "map",
      resume: { kind: null, reason: "engine mismatch" } }));
    expect(cmds["play-from-here"].disabledReason).toBe("engine mismatch");
    const fixture = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, stepDecision: "map",
      resume: { kind: "combat", reason: null } }));
    expect(fixture["play-from-here"].disabledReason).toMatch(/combat fixture/);
    const ok = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, stepDecision: "combat_play",
      resume: { kind: "combat", reason: null } }));
    expect(ok["play-from-here"].disabledReason).toBeUndefined();
    // No resume capability reported at all → the command is not offered.
    expect(byId(ctx({ mode: "replay", hasLive: false, replayCount: 5 }))["play-from-here"]).toBeUndefined();
  });

  it("offers the live-game bridge commands only in replay with the bridge up", () => {
    const off = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5 }));
    expect(off["bridge-tree"]).toBeUndefined();
    const on = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, bridgeConnected: true, viewedFloor: 7 }));
    expect(on["bridge-tree"]).toBeDefined();
    expect(on["bridge-seek"].label).toContain("floor 7");
    const noFloor = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, bridgeConnected: true }));
    expect(noFloor["bridge-seek"]).toBeUndefined();
  });

  it("marks busy commands and runs the wired action", () => {
    const recommend = vi.fn();
    const cmds = byId(ctx({ mode: "replay", hasLive: false, replayCount: 5, stepDecision: "combat_play",
      busy: { recommend: true }, actions: { ...ctx().actions, recommend } }));
    expect(cmds["recommend"].busy).toBe(true);
    cmds["recommend"].run();
    expect(recommend).toHaveBeenCalledOnce();
  });

  it("gives every command a unique id and a known group", () => {
    const all = buildCommands(ctx({ mode: "replay", hasLive: false, replayCount: 5, bridgeConnected: true, viewedFloor: 1 }));
    expect(new Set(all.map((c) => c.id)).size).toBe(all.length);
    for (const c of all) expect(["run", "analyze"]).toContain(c.group);
  });
});

describe("the live-game views in the browser build (GH #709)", () => {
  it("show in any view once the page talks to the control plane directly, but floor-seek stays a replay command", () => {
    const direct = byId(ctx({ mode: "play", hasLive: true, bridgeConnected: true, bridgeDirect: true, viewedFloor: 3 }));
    expect(direct["bridge-tree"]).toBeDefined();
    expect(direct["live-search"]).toBeDefined();
    expect(direct["bridge-seek"]).toBeUndefined();
    // Configured but not connected: nothing.
    const off = byId(ctx({ mode: "play", hasLive: true, bridgeConnected: false, bridgeDirect: true }));
    expect(off["bridge-tree"]).toBeUndefined();
  });
});
