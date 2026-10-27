//! Golden parity against the Python workbench (the oracle).
//!
//! `tests/golden/<scenario>/requests.jsonl` was recorded by
//! `sts2-pycli/scripts/dump_workbench_golden.py` driving the FastAPI app
//! through scripted play; each line is one request plus the SHA-256 of the
//! NORMALIZED response body. This test replays the same requests through
//! [`sts2_workbench::api::Workbench`] and demands the same status and hash.
//! Where the dump also stored the full body (`responses/<i>.json`) a
//! mismatch is reported as a structural diff; otherwise as the hash and the
//! first differing key path is unavailable — re-dump with the Python script
//! to get the body.
//!
//! The advisor probes (M2) come in two kinds. The search-backed routes —
//! hint, search tree, choice ranker — are one seeded search over the same
//! engine state through the same emitter on both hosts, so they are hashed
//! like everything else. The routes in [`STRUCTURAL_ROUTES`] cannot match by
//! hash on purpose: the regret routes re-score the parent's NATIVE snapshot
//! where Python re-scores a lossy wire dict, and "play best" off combat asks
//! the Rust `HeuristicPolicy` where Python asks its own `run_policy`. Those
//! are held to status + top-level key shape (when the body was stored).
//!
//! Run mode (M3): a scenario whose `meta.json` carries `scenario.ppo` served
//! the committed champion on the Python side; this test loads the same ONNX
//! bytes from the sibling checkout before replaying it, so the panel
//! probabilities, node values and the WPA review hash-match (tract, same
//! bytes, both hosts). Without a net the `ppo` brain falls back to each
//! host's own heuristic (see above), so `autobattle/verdict` — which grades
//! against the brain's pick — is structural in those scenarios only.
//!
//! The replay viewer (M4): the dump ends every scenario with a quit and
//! drives the replay routes over the log it left. The host side here mirrors
//! the worker: the log `take_run_log` hands back is kept under its filename
//! and `/api/load` finds it by the recorded path's basename (the Python side
//! wrote to a temp dir; both stamp the same pinned clock). `/api/runs` and
//! `/api/runs/stat` carry host paths and mtimes, and the replay's
//! `recommend` / `search_tree` / `regret-review` search the exact engine
//! state rebuilt from the recorded line where Python re-scores a lossy
//! reconstruction — all structural.
//!
//! Normalization (identical on both sides — see the script's docstring):
//! sort object keys recursively, drop the keys `log_path` and `path`
//! anywhere, round every float to 4 decimals, hash the compact JSON
//! (`separators=(",", ":")`, non-ASCII kept raw).

// The recorded goldens carry rendered game text and only replay against the
// real bundle (`data/game/`, see build.rs) — without it this crate is empty.
#![cfg(workbench_game_data)]

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use serde_json::{json, Map, Value};
use sha2::{Digest, Sha256};
use sts2_workbench::api::{ApiError, Config, Workbench};

const ASSETS_BASE: &str = "https://assets.test";

/// Routes compared by status + key shape instead of body hash (see the
/// module docs for why each cannot be bit-exact).
const STRUCTURAL_ROUTES: &[&str] = &[
    "/api/live/auto-action",
    "/api/live/last-move-regret",
    "/api/live/regret-review",
    // M4 replay: host paths / mtimes, and the exact-state re-searches.
    "/api/runs",
    "/api/runs/stat",
    "/api/recommend",
    "/api/search_tree",
    "/api/regret-review",
];

/// The route without its query string.
fn route_path(route: &str) -> &str {
    route.split('?').next().unwrap_or(route)
}

/// Top-level keys of an object (a non-object has none).
fn key_shape(v: &Value) -> Vec<String> {
    v.as_object()
        .map(|m| m.keys().cloned().collect())
        .unwrap_or_default()
}
const SAVED_AT: &str = "2026-01-01T00:00:00Z";

fn golden_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/golden")
}

/// The dump script's normalization.
fn normalize(v: &Value) -> Value {
    match v {
        Value::Object(m) => {
            let mut out = Map::new();
            for (k, val) in m {
                if k == "log_path" || k == "path" {
                    continue;
                }
                out.insert(k.clone(), normalize(val));
            }
            Value::Object(out)
        }
        Value::Array(a) => Value::Array(a.iter().map(normalize).collect()),
        Value::Number(n) => {
            if n.is_f64() {
                let f = n.as_f64().unwrap_or(0.0);
                serde_json::Number::from_f64(py_round4(f)).map_or(Value::Null, Value::Number)
            } else {
                v.clone()
            }
        }
        other => other.clone(),
    }
}

/// Python `round(x, 4)`: correctly rounded on the EXACT binary value, ties
/// to even. `(x * 1e4).round() / 1e4` is not that — the scaled product picks
/// up its own rounding error, so a path-plan score whose f64 sits a hair
/// below 0.75625 scaled to exactly 7562.5 and went up where Python went down
/// (`long_ironclad_g0lden0001_a0_roll` #190). Rust's `{:.4}` formatting is
/// exact and ties-to-even, and Python's `round` returns the float nearest the
/// resulting decimal, which is what parsing it back gives.
fn py_round4(f: f64) -> f64 {
    format!("{f:.4}").parse().unwrap_or(f)
}

/// `json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)`
/// for an already-normalized value (serde_json's `Map` is sorted).
fn canonical(v: &Value) -> String {
    serde_json::to_string(v).expect("serialize")
}

fn sha256_hex(s: &str) -> String {
    use std::fmt::Write as _;
    let digest = Sha256::digest(s.as_bytes());
    digest.iter().fold(String::with_capacity(64), |mut out, b| {
        let _ = write!(out, "{b:02x}");
        out
    })
}

/// Python `slugify` (saves.py) for the in-memory save store.
fn slugify(name: &str) -> String {
    sts2_workbench::saves::slugify(name)
}

/// One recorded request.
struct Req {
    i: u64,
    method: String,
    route: String,
    body: Value,
    status: u16,
    sha256: String,
}

fn load_requests(dir: &Path) -> Vec<Req> {
    let text = fs::read_to_string(dir.join("requests.jsonl")).expect("requests.jsonl");
    text.lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Value = serde_json::from_str(l).expect("request line");
            Req {
                i: v["i"].as_u64().expect("i"),
                method: v["method"].as_str().expect("method").to_string(),
                route: v["route"].as_str().expect("route").to_string(),
                body: v.get("body").cloned().unwrap_or(Value::Null),
                status: u16::try_from(v["status"].as_u64().expect("status")).expect("status u16"),
                sha256: v["sha256"].as_str().expect("sha256").to_string(),
            }
        })
        .collect()
}

/// The host side of the save + run-log routes, exactly as the worker
/// implements them.
struct SaveStore {
    saves: BTreeMap<String, Value>,
    /// Run logs by filename (`take_run_log`), newest last.
    runs: Vec<(String, String)>,
}

impl SaveStore {
    fn route(&mut self, wb: &mut Workbench, req: &Req) -> Option<Result<Value, ApiError>> {
        match (req.method.as_str(), route_path(&req.route)) {
            ("POST", "/api/live/quit") => {
                let out = wb.call("POST", "/api/live/quit", &req.body);
                if let Some(log) = wb.take_run_log() {
                    self.runs.push((log.filename, log.text));
                }
                Some(out)
            }
            ("GET", "/api/runs") => {
                let mut names: Vec<&String> = self.runs.iter().map(|(n, _)| n).collect();
                names.reverse();
                Some(Ok(json!({"runs": names})))
            }
            ("GET", "/api/runs/stat") => Some(Ok(json!({"size": 0, "mtime": 0.0}))),
            ("POST", "/api/load") => {
                let path = req.body.get("path").and_then(Value::as_str).unwrap_or("");
                let base = path.rsplit('/').next().unwrap_or(path);
                Some(match self.runs.iter().find(|(n, _)| n == base) {
                    Some((name, text)) => wb.load_run(name, text),
                    None => Err(ApiError::new(404, format!("file not found: {path}"))),
                })
            }
            ("GET", "/api/mcts-runs") => Some(Ok(Workbench::mcts_runs(&self.runs))),
            ("GET", "/api/mcts-runs/aggregate") => Some(Ok(Workbench::aggregate_runs(&self.runs))),
            ("POST", "/api/live/save") => {
                let name = req.body.get("name").and_then(Value::as_str).unwrap_or("");
                Some(wb.build_save(name, SAVED_AT).map(|payload| {
                    let step_count = payload.get("step_count").cloned().unwrap_or(json!(0));
                    self.saves.insert(slugify(name), payload);
                    json!({"name": name, "path": Value::Null, "step_count": step_count})
                }))
            }
            ("GET", "/api/saves") => {
                let mut headers: Vec<Value> =
                    self.saves.values().map(Workbench::save_header).collect();
                headers.sort_by(|a, b| {
                    let ka = a.get("saved_at").and_then(Value::as_str).unwrap_or("");
                    let kb = b.get("saved_at").and_then(Value::as_str).unwrap_or("");
                    kb.cmp(ka)
                });
                Some(Ok(json!({"saves": headers})))
            }
            ("POST", "/api/saves/load") => {
                let name = req.body.get("name").and_then(Value::as_str).unwrap_or("");
                Some(match self.saves.get(&slugify(name)) {
                    None => Err(ApiError::new(404, format!("unknown save: {name}"))),
                    Some(payload) => wb.restore_save(&payload.clone()),
                })
            }
            _ => None,
        }
    }
}

/// First differing JSON path between two normalized values.
fn first_diff(a: &Value, b: &Value, path: &str) -> Option<String> {
    match (a, b) {
        (Value::Object(x), Value::Object(y)) => {
            for k in x.keys().chain(y.keys()) {
                match (x.get(k), y.get(k)) {
                    (Some(_), None) => return Some(format!("{path}.{k}: only in python")),
                    (None, Some(_)) => return Some(format!("{path}.{k}: only in rust")),
                    (Some(xv), Some(yv)) => {
                        if let Some(d) = first_diff(xv, yv, &format!("{path}.{k}")) {
                            return Some(d);
                        }
                    }
                    (None, None) => {}
                }
            }
            None
        }
        (Value::Array(x), Value::Array(y)) => {
            if x.len() != y.len() {
                return Some(format!("{path}: len python={} rust={}", x.len(), y.len()));
            }
            for (i, (xv, yv)) in x.iter().zip(y).enumerate() {
                if let Some(d) = first_diff(xv, yv, &format!("{path}[{i}]")) {
                    return Some(d);
                }
            }
            None
        }
        _ => {
            if a == b {
                None
            } else {
                let show = |v: &Value| {
                    let s = v.to_string();
                    if s.len() > 200 {
                        format!("{}…", &s[..200])
                    } else {
                        s
                    }
                };
                Some(format!("{path}: python={} rust={}", show(a), show(b)))
            }
        }
    }
}

/// `meta.json`'s `scenario` block (the dump's input for this fixture).
fn scenario_meta(dir: &Path) -> Value {
    fs::read_to_string(dir.join("meta.json"))
        .ok()
        .and_then(|t| serde_json::from_str::<Value>(&t).ok())
        .and_then(|m| m.get("scenario").cloned())
        .unwrap_or(Value::Null)
}

/// The committed champion the dump served (`sts2-pycli/data/ppo/<file>`).
fn champion_bytes(character: &str) -> Vec<u8> {
    let file = Workbench::ppo_champion_file(character).expect("champion file");
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../../sts2-pycli/data/ppo")
        .join(file);
    fs::read(&path).unwrap_or_else(|e| panic!("champion {} missing: {e}", path.display()))
}

fn replay_scenario(dir: &Path) -> Vec<String> {
    let mut wb = Workbench::new(Config {
        assets_base: ASSETS_BASE.to_string(),
        ..Config::default()
    });
    let scenario = scenario_meta(dir);
    let ppo = scenario["ppo"].as_bool().unwrap_or(false);
    if ppo {
        let character = scenario["character"].as_str().expect("character");
        wb.load_ppo(character, &champion_bytes(character))
            .expect("load champion");
    }
    // Without a net the brain's pick is each host's own heuristic.
    let structural_verdict = !ppo;
    let mut store = SaveStore {
        saves: BTreeMap::new(),
        runs: Vec::new(),
    };
    let mut failures = Vec::new();
    for req in load_requests(dir) {
        let result = store
            .route(&mut wb, &req)
            .unwrap_or_else(|| wb.call(&req.method, &req.route, &req.body));
        let (status, body) = match result {
            Ok(b) => (200_u16, b),
            Err(e) => (e.status, e.body()),
        };
        let norm = normalize(&body);
        let hash = sha256_hex(&canonical(&norm));
        let structural = STRUCTURAL_ROUTES.contains(&route_path(&req.route))
            || (structural_verdict && req.route == "/api/live/autobattle/verdict");
        if structural {
            let stored = dir.join("responses").join(format!("{}.json", req.i));
            let py_keys = if stored.is_file() {
                let py: Value = serde_json::from_str(&fs::read_to_string(&stored).expect("stored"))
                    .expect("stored json");
                Some(key_shape(&normalize(&py)))
            } else {
                None
            };
            let rust_keys = key_shape(&norm);
            let (status_ok, shape_ok) = match req.route.as_str() {
                // Python's off-combat brain tags its answer (`policy`,
                // `fallback`); the heuristic answer here is the bare verb.
                "/api/live/auto-action" => (
                    status == req.status,
                    py_keys
                        .as_ref()
                        .is_none_or(|py| rust_keys.iter().all(|k| py.contains(k))),
                ),
                // The coach assesses every move the oracle does and more:
                // Python's wire round-trip loses a potion's slot and can lose
                // the played card, and then answers 409 where the native
                // snapshot scores the move. Never the reverse.
                "/api/live/last-move-regret" => (
                    status == req.status || (req.status == 409 && status == 200),
                    status != req.status || py_keys.as_ref().is_none_or(|py| *py == rust_keys),
                ),
                _ => (
                    status == req.status,
                    py_keys.as_ref().is_none_or(|py| *py == rust_keys),
                ),
            };
            if !status_ok || !shape_ok {
                failures.push(format!(
                    "#{} {} {}: status python={} rust={}; keys rust={:?}",
                    req.i,
                    req.method,
                    req.route,
                    req.status,
                    status,
                    key_shape(&norm)
                ));
            }
            continue;
        }
        if status != req.status || hash != req.sha256 {
            let stored = dir.join("responses").join(format!("{}.json", req.i));
            let detail = if stored.is_file() {
                let py: Value = serde_json::from_str(&fs::read_to_string(&stored).expect("stored"))
                    .expect("stored json");
                first_diff(&normalize(&py), &norm, "$").unwrap_or_else(|| {
                    "(bodies equal, hash differs — canonical serialization mismatch)".to_string()
                })
            } else {
                "(no stored body — re-dump with `just workbench-golden-dump` to see the diff)"
                    .to_string()
            };
            failures.push(format!(
                "#{} {} {}: status python={} rust={}; {}",
                req.i, req.method, req.route, req.status, status, detail
            ));
            if failures.len() >= 3 {
                failures.push(
                    "(stopping after 3 divergences; later steps replay from a diverged state)"
                        .to_string(),
                );
                break;
            }
        }
    }
    failures
}

fn scenario_dirs() -> Vec<PathBuf> {
    let root = golden_root();
    let Ok(rd) = fs::read_dir(&root) else {
        return Vec::new();
    };
    let mut dirs: Vec<PathBuf> = rd
        .filter_map(Result::ok)
        .map(|e| e.path())
        .filter(|p| p.join("requests.jsonl").is_file())
        .filter(|p| build_bundle_present(p))
        .collect();
    dirs.sort();
    dirs
}

/// False for a scenario played in a game build whose display bundle this
/// checkout lacks (the public snapshot ships only the default build's): its
/// responses would render loc keys, so it cannot match its recording (#855).
fn build_bundle_present(dir: &Path) -> bool {
    let first = fs::read_to_string(dir.join("requests.jsonl"))
        .ok()
        .and_then(|t| t.lines().next().map(str::to_string))
        .and_then(|l| serde_json::from_str::<Value>(&l).ok());
    let version = first
        .as_ref()
        .and_then(|r| r["body"]["game_version"].as_str())
        .unwrap_or_default();
    let present = version != "v0.111.0" || cfg!(workbench_game_data_v0_111_0);
    if !present {
        eprintln!(
            "skip {}: no {version} workbench bundle in this checkout",
            dir.display()
        );
    }
    present
}

#[test]
fn golden_fixtures_are_present() {
    let dirs = scenario_dirs();
    assert!(
        !dirs.is_empty(),
        "no golden fixtures under {} — run `just workbench-golden-dump` from sts2-pycli/",
        golden_root().display()
    );
}

/// Replay every golden scenario, in parallel across OS threads.
///
/// ## Why this is threaded rather than a plain loop
///
/// Serially this is ~20s in release but over **180s in a debug build**, which
/// is `.config/nextest.toml`'s `slow-timeout` × `terminate-after`. A timed-out
/// test is not a failing test: nextest reports TIMEOUT, which reads like a
/// busy machine rather than "your change broke a golden", and `cargo nextest
/// run --workspace` can never go green — on a repo whose local gates are the
/// only gates (see the root `CLAUDE.md`: there is deliberately no CI).
///
/// That is not hypothetical. Promoting the Silent champion
/// (`ppo_fullrun_silent_cont_it300`, #700) genuinely diverged
/// `runmode_silent_run2_a0` — its recorded PPO probabilities belonged to the
/// previous net — and the debug run died before reaching it. The divergence
/// surfaced only because the suite was re-run in release. See GH #701.
///
/// [`replay_scenario`] builds its own [`Workbench`] and touches no shared
/// state, so the scenarios are independent by construction; the only shared
/// thing is the read-only fixture tree. Results are re-sorted into directory
/// order before reporting, so the message is deterministic regardless of which
/// thread finished first.
fn replay_all_scenarios(dirs: &[PathBuf]) -> Vec<(usize, String)> {
    // Each worker holds a Workbench and, for the `ppo` scenarios, a tract net,
    // so this is capped below the core count on purpose. This test is the
    // nextest run's long pole, so the cap bounds the gate's wall clock.
    let threads = std::thread::available_parallelism()
        .map_or(4, |n| n.get().min(16))
        .min(dirs.len().max(1));
    // Longest scenario first (request-log size as the proxy). Handed out in
    // name order, `long_defect_*` (2x the next one) started late and ran alone.
    let mut order: Vec<usize> = (0..dirs.len()).collect();
    order.sort_by_cached_key(|&i| {
        std::cmp::Reverse(fs::metadata(dirs[i].join("requests.jsonl")).map_or(0, |m| m.len()))
    });
    let next = std::sync::atomic::AtomicUsize::new(0);
    let out = std::sync::Mutex::new(Vec::new());

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| loop {
                let k = next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                let Some(&i) = order.get(k) else { return };
                let dir = &dirs[i];
                let name = dir
                    .file_name()
                    .and_then(|n| n.to_str())
                    .unwrap_or("?")
                    .to_string();
                let failures = replay_scenario(dir);
                if !failures.is_empty() {
                    out.lock()
                        .expect("report mutex")
                        .push((i, format!("== {name}\n{}", failures.join("\n"))));
                }
            });
        }
    });

    let mut report = out.into_inner().expect("report mutex");
    report.sort_by_key(|(i, _)| *i);
    report
}

#[test]
fn golden_parity_all_scenarios() {
    let dirs = scenario_dirs();
    if dirs.is_empty() {
        eprintln!("skip: no golden fixtures recorded yet");
        return;
    }
    let report = replay_all_scenarios(&dirs);
    assert!(
        report.is_empty(),
        "{} of {} scenarios diverged from the Python oracle:\n{}",
        report.len(),
        dirs.len(),
        report
            .iter()
            .map(|(_, r)| r.as_str())
            .collect::<Vec<_>>()
            .join("\n\n")
    );
}

#[test]
fn normalization_matches_the_dump_contract() {
    let v = json!({"b": 1.23456, "a": [{"path": "x", "log_path": null, "z": 2}], "c": "é\n"});
    let n = normalize(&v);
    assert_eq!(
        canonical(&n),
        "{\"a\":[{\"z\":2}],\"b\":1.2346,\"c\":\"é\\n\"}"
    );
    assert_eq!(
        sha256_hex("abc"),
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    );
}
