//! Deterministic, dependency-free content hashing of a crate's build inputs.
//!
//! Issue #15 / bead `sts2-rs-cg5ci`. The parity harness must be able to prove
//! that the loaded `sts2_rs` `.so` was compiled from exactly the source it
//! claims to test — not an mtime, not a git SHA (both have blind spots), but
//! the source *bytes*. This crate is the single hash implementation used by
//! BOTH sides of that proof:
//!
//!   * each engine crate's `build.rs` calls [`hash_crate`] (or
//!     [`emit_build_stamp`]) over its own source and bakes the result into a
//!     `pub const SRC_HASH` compiled INTO the crate (so a crate cargo wrongly
//!     skips recompiling carries its *old* hash), and
//!   * `sts2-py` exposes a runtime recompute over the on-disk source.
//!
//! Because both call the SAME function here, the two can never drift — the
//! "one hasher" rule from the bead. The hash only needs to be deterministic
//! for given bytes; it is NOT cryptographic (staleness detection, not defense),
//! so a dependency-free FNV-1a keeps `build.rs` free of build-dependencies.

use std::path::{Path, PathBuf};

const FNV_OFFSET: u64 = 0xcbf2_9ce4_8422_2325;
const FNV_PRIME: u64 = 0x0000_0100_0000_01b3;

/// FNV-1a 64-bit, folding `bytes` into the running accumulator `h`.
fn fnv1a(mut h: u64, bytes: &[u8]) -> u64 {
    for &b in bytes {
        h ^= u64::from(b);
        h = h.wrapping_mul(FNV_PRIME);
    }
    h
}

/// Hash a byte slice to a 16-char lowercase-hex string (FNV-1a 64-bit).
pub fn hash_bytes(bytes: &[u8]) -> String {
    format!("{:016x}", fnv1a(FNV_OFFSET, bytes))
}

/// Hash a crate's own build inputs into a 16-char hex string: every file under
/// `<crate_dir>/src`, the crate's `Cargo.toml`, and any `extra` inputs (e.g. an
/// `include_bytes!` data file living outside `src`). `extra` is `(label, path)`
/// — the LABEL (not the path) is hashed, so the same content under a different
/// absolute path (canonical checkout vs worktree) yields the SAME hash, which
/// is what lets the build-time stamp and the runtime recompute match.
pub fn hash_crate(crate_dir: &Path, extra: &[(&str, PathBuf)]) -> String {
    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    collect_dir(&crate_dir.join("src"), "src", &mut records);
    if let Ok(bytes) = std::fs::read(crate_dir.join("Cargo.toml")) {
        records.push(("Cargo.toml".to_string(), bytes));
    }
    for (label, path) in extra {
        if let Ok(bytes) = std::fs::read(path) {
            records.push(((*label).to_string(), bytes));
        }
    }
    digest(records)
}

/// Sort `records` by label, then fold them into a length-framed FNV-1a digest.
/// Length-framing means neither a label nor a content blob that happens to
/// contain a separator byte can be mistaken for a record boundary.
fn digest(mut records: Vec<(String, Vec<u8>)>) -> String {
    records.sort_by(|a, b| a.0.cmp(&b.0));
    let mut h = FNV_OFFSET;
    for (label, content) in &records {
        h = fnv1a(h, label.as_bytes());
        h = fnv1a(h, &[0]);
        h = fnv1a(
            h,
            &u64::try_from(content.len())
                .unwrap_or(u64::MAX)
                .to_le_bytes(),
        );
        h = fnv1a(h, content);
        h = fnv1a(h, &[0xff]);
    }
    format!("{h:016x}")
}

/// Combine component hashes, in the given order, into one aggregate hash.
/// Order-sensitive by design (callers pass a fixed crate order).
pub fn combine(parts: &[&str]) -> String {
    let mut h = FNV_OFFSET;
    for p in parts {
        h = fnv1a(h, p.as_bytes());
        h = fnv1a(h, &[0]);
    }
    format!("{h:016x}")
}

/// Extra out-of-`src` engine inputs for sts2-py, as `(label, relative path)`.
///
/// **Currently empty.** It used to carry
/// `data/value_policy_net/embed_policy_prior.json`, the `include_bytes!`-baked
/// policy prior; that prior was deleted on 2026-08-15
/// (`logs/2026-08-15-policy-prior/FINDINGS.md`) and sts2-py no longer has any
/// data input outside its own `src`. The mechanism is kept — `hash_crate`'s
/// `extra` parameter is still exercised by this crate's tests — because the
/// next `include_bytes!` asset must be hashed the same way, and an unhashed
/// one would let two different engines share a content identity.
pub const PY_DATA_INPUTS: [(&str, &str); 0] = [];

/// Extra out-of-`src` engine inputs for sts2-runai: the HeuristicV weights
/// `src/eval/mod.rs` bakes in with `include_str!`.
pub const RUNAI_DATA_INPUTS: [(&str, &str); 1] =
    [("data/runai/heuristic_v.toml", "data/runai/heuristic_v.toml")];

/// Extra out-of-`src` engine inputs for sts2-mcts: the built-in playout nets
/// `src/rollout_net.rs` bakes in with `include_str!`, one per character, and
/// the built-in Ironclad move prior (`data/prior_nets`, GH #830) it bakes in
/// the same way. A missing input is skipped, as everywhere. An entry here is
/// also what makes `build.rs` rerun, so editing or adding a file flips its
/// `cfg` and moves `src_hash` without a `cargo clean`.
pub const MCTS_DATA_INPUTS: [(&str, &str); 6] = [
    (
        "data/rollout_nets/ironclad.json",
        "data/rollout_nets/ironclad.json",
    ),
    (
        "data/rollout_nets/silent.json",
        "data/rollout_nets/silent.json",
    ),
    (
        "data/rollout_nets/defect.json",
        "data/rollout_nets/defect.json",
    ),
    (
        "data/rollout_nets/regent.json",
        "data/rollout_nets/regent.json",
    ),
    (
        "data/rollout_nets/necrobinder.json",
        "data/rollout_nets/necrobinder.json",
    ),
    (
        "data/prior_nets/ironclad.json",
        "data/prior_nets/ironclad.json",
    ),
];

/// Every workspace crate compiled into the `sts2_rs` extension, in the FIXED
/// order used by both the aggregate and the recompute. `sts2_py::src_hash()`
/// must combine the compiled-in consts in this same order.
///
/// That is the closure of sts2-py's normal path dependencies, minus
/// `sts2-srchash` itself (it only computes these hashes). It used to be the
/// first four alone, so an edit to the run-AI, wire, CLI or vocab crate left
/// `src_hash` unchanged: a `.so` without the edit passed every freshness check
/// (the PPO trainer's guard, `wheel-venv.sh`'s keep-the-PGO-wheel path). The
/// sts2-py test `engine_crates_are_what_sts2_rs_links` checks the list against
/// `cargo metadata`, so a new dependency cannot be left out again.
pub const ENGINE_CRATES: [&str; 8] = [
    "sts2-core",
    "sts2-content",
    "sts2-mcts",
    "sts2-py",
    "sts2-runai",
    "sts2-wire",
    "sts2-cli",
    "sts2-vocab",
];

/// The out-of-`src` inputs of engine crate `krate`, as `(label, path relative
/// to the sts2-rs root)`.
pub fn data_inputs(krate: &str) -> &'static [(&'static str, &'static str)] {
    match krate {
        "sts2-py" => &PY_DATA_INPUTS,
        "sts2-runai" => &RUNAI_DATA_INPUTS,
        "sts2-mcts" => &MCTS_DATA_INPUTS,
        _ => &[],
    }
}

/// [`hash_crate`] of engine crate `krate` in the checkout at `rs_root`, with its
/// [`data_inputs`].
pub fn hash_engine_crate(rs_root: &Path, krate: &str) -> String {
    let extra: Vec<(&str, PathBuf)> = data_inputs(krate)
        .iter()
        .map(|(label, rel)| (*label, rs_root.join(rel)))
        .collect();
    hash_crate(&rs_root.join("crates").join(krate), &extra)
}

/// The build-script half of [`hash_engine_crate`], for an engine crate's
/// `build.rs`: prints `cargo:rustc-env=<env_var>=<hash>` and a
/// `rerun-if-changed` for every input it hashed (`src`, `Cargo.toml` and the
/// [`data_inputs`]). Cargo's default rerun rule (any file in the package) would
/// miss a data input outside the package and leave the stamp stale.
pub fn emit_build_stamp(krate: &str, env_var: &str) {
    let crate_dir = PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"),
    );
    assert_eq!(
        crate_dir.file_name().and_then(|n| n.to_str()),
        Some(krate),
        "emit_build_stamp({krate:?}) called from another crate's build script"
    );
    let rs_root = crate_dir
        .parent()
        .and_then(Path::parent)
        .expect("an engine crate lives at <rs_root>/crates/<name>");
    println!(
        "cargo:rustc-env={env_var}={}",
        hash_engine_crate(rs_root, krate)
    );
    let data = data_inputs(krate)
        .iter()
        .map(|(_, rel)| rerun_watch_path(rs_root, &rs_root.join(rel)));
    for path in [crate_dir.join("src"), crate_dir.join("Cargo.toml")]
        .into_iter()
        .chain(data)
    {
        println!("cargo:rerun-if-changed={}", path.display());
    }
}

/// What to name in `rerun-if-changed` for data input `path`: the file when it
/// exists, else its nearest existing ancestor directory (no higher than
/// `rs_root`). Cargo treats a missing `rerun-if-changed` path as always
/// stale, so naming an absent optional input would rerun the build script —
/// and rebuild every dependent crate — on every build. A directory is scanned
/// for changes, so the input appearing later still triggers the rerun.
fn rerun_watch_path(rs_root: &Path, path: &Path) -> PathBuf {
    let mut p = path;
    while !p.exists() && p != rs_root {
        match p.parent() {
            Some(parent) => p = parent,
            None => break,
        }
    }
    p.to_path_buf()
}

/// Hash the workspace-wide engine inputs that no single crate's `src` owns: the
/// workspace manifest (`[profile.*]`, `[workspace.dependencies]` features), the
/// resolved dependency versions (`Cargo.lock`) and the pinned toolchain
/// (`rust-toolchain.toml`). `rs_root` is the sts2-rs checkout root.
pub fn hash_workspace(rs_root: &Path) -> String {
    let mut records: Vec<(String, Vec<u8>)> = Vec::new();
    for name in ["Cargo.toml", "Cargo.lock", "rust-toolchain.toml"] {
        if let Ok(bytes) = std::fs::read(rs_root.join(name)) {
            records.push((name.to_string(), bytes));
        }
    }
    digest(records)
}

/// Recompute the aggregate engine hash from the on-disk source at `rs_root`
/// (the sts2-rs checkout root containing `crates/`, `data/`, `Cargo.lock`).
/// Combines the engine crates' per-crate hashes and the workspace hash in
/// [`ENGINE_CRATES`] order — the SAME order `sts2_py::src_hash()` combines the
/// compiled-in `SRC_HASH` consts — so on a clean tree baked == recomputed. This
/// is the runtime half of the content-identity proof (issue #15).
pub fn recompute_engine_hash(rs_root: &Path) -> String {
    let mut parts: Vec<String> = ENGINE_CRATES
        .iter()
        .map(|krate| hash_engine_crate(rs_root, krate))
        .collect();
    parts.push(hash_workspace(rs_root));
    combine(&parts.iter().map(String::as_str).collect::<Vec<_>>())
}

/// Recursively collect every file under `dir` as `(label, contents)`, where
/// `label` is the slash-joined path relative to the original root (prefixed by
/// `prefix`). A missing/unreadable dir or file is skipped — `hash_crate` reads
/// `src` + `Cargo.toml` explicitly, so "src absent" simply contributes nothing.
fn collect_dir(dir: &Path, prefix: &str, out: &mut Vec<(String, Vec<u8>)>) {
    let Ok(entries) = std::fs::read_dir(dir) else {
        return;
    };
    for entry in entries.flatten() {
        let name = entry.file_name().to_string_lossy().into_owned();
        let label = if prefix.is_empty() {
            name
        } else {
            format!("{prefix}/{name}")
        };
        match entry.file_type() {
            Ok(ft) if ft.is_dir() => collect_dir(&entry.path(), &label, out),
            Ok(ft) if ft.is_file() => {
                if let Ok(bytes) = std::fs::read(entry.path()) {
                    out.push((label, bytes));
                }
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;
    use std::sync::atomic::{AtomicUsize, Ordering};

    static CTR: AtomicUsize = AtomicUsize::new(0);

    /// A fresh temp crate dir with a `src/` and a minimal `Cargo.toml`. Unique
    /// per call (pid + counter) — no Date/random needed.
    fn tmp_crate() -> PathBuf {
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let d = std::env::temp_dir().join(format!("sts2srchash_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(d.join("src")).unwrap();
        fs::write(d.join("Cargo.toml"), b"[package]\nname=\"x\"\n").unwrap();
        d
    }

    /// A fake sts2-rs checkout root: every engine crate dir, the runai weights,
    /// the Ironclad move prior, the workspace `Cargo.toml`, `Cargo.lock`, and
    /// `rust-toolchain.toml`.
    fn tmp_rs_root() -> PathBuf {
        let n = CTR.fetch_add(1, Ordering::Relaxed);
        let root = std::env::temp_dir().join(format!("sts2rsroot_{}_{n}", std::process::id()));
        let _ = fs::remove_dir_all(&root);
        for c in ENGINE_CRATES {
            let cd = root.join("crates").join(c);
            fs::create_dir_all(cd.join("src")).unwrap();
            fs::write(cd.join("Cargo.toml"), b"[package]\n").unwrap();
            fs::write(cd.join("src/lib.rs"), b"// x").unwrap();
        }
        fs::create_dir_all(root.join("data/runai")).unwrap();
        fs::write(root.join("data/runai/heuristic_v.toml"), b"w = 1").unwrap();
        fs::create_dir_all(root.join("data/prior_nets")).unwrap();
        fs::write(root.join("data/prior_nets/ironclad.json"), b"{}").unwrap();
        fs::write(root.join("Cargo.toml"), b"[workspace]\n").unwrap();
        fs::write(root.join("Cargo.lock"), b"lock-v1").unwrap();
        fs::write(
            root.join("rust-toolchain.toml"),
            b"[toolchain]\nchannel=\"1.95\"\n",
        )
        .unwrap();
        root
    }

    #[test]
    fn recompute_engine_hash_deterministic_and_sensitive_to_a_crate() {
        let root = tmp_rs_root();
        let h1 = recompute_engine_hash(&root);
        assert_eq!(h1, recompute_engine_hash(&root));
        assert_eq!(h1.len(), 16);
        fs::write(root.join("crates/sts2-core/src/lib.rs"), b"// edited").unwrap();
        assert_ne!(
            h1,
            recompute_engine_hash(&root),
            "a core src edit must change the aggregate"
        );
    }

    /// Every crate compiled into the wheel moves the aggregate, not just the
    /// engine core: the PPO trainer's combat agent lives in sts2-runai. So do
    /// the compiled-in data files: runai's HeuristicV weights and the
    /// Ironclad move prior (GH #830), whose edit must move `src_hash` and
    /// rebuild without a `cargo clean`.
    #[test]
    fn recompute_engine_hash_tracks_every_linked_crate_and_its_data() {
        let root = tmp_rs_root();
        let mut seen = vec![recompute_engine_hash(&root)];
        for c in ENGINE_CRATES {
            fs::write(root.join("crates").join(c).join("src/lib.rs"), b"// edited").unwrap();
            seen.push(recompute_engine_hash(&root));
        }
        fs::write(root.join("data/runai/heuristic_v.toml"), b"w = 2").unwrap();
        seen.push(recompute_engine_hash(&root));
        fs::write(root.join("data/prior_nets/ironclad.json"), b"{\"meta\":{}}").unwrap();
        seen.push(recompute_engine_hash(&root));
        let distinct: std::collections::HashSet<_> = seen.iter().collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "an edit left the hash unchanged: {seen:?}"
        );
    }

    #[test]
    fn recompute_engine_hash_tracks_workspace() {
        let root = tmp_rs_root();
        let mut seen = vec![recompute_engine_hash(&root)];
        for (name, bytes) in [
            ("Cargo.toml", &b"[workspace]\n[profile.release]\n"[..]),
            ("Cargo.lock", b"lock-v2"),
            ("rust-toolchain.toml", b"[toolchain]\nchannel=\"1.96\"\n"),
        ] {
            fs::write(root.join(name), bytes).unwrap();
            seen.push(recompute_engine_hash(&root));
        }
        let distinct: std::collections::HashSet<_> = seen.iter().collect();
        assert_eq!(
            distinct.len(),
            seen.len(),
            "a workspace manifest, Cargo.lock or toolchain change must change the aggregate: {seen:?}"
        );
    }

    /// `PY_DATA_INPUTS` is empty today, so no test can observe an out-of-`src`
    /// data input through `recompute_engine_hash`. Exercise the mechanism
    /// directly instead, so the next `include_bytes!` asset inherits working,
    /// tested plumbing rather than plumbing that quietly stopped being covered
    /// when the policy prior was deleted.
    #[test]
    fn hash_crate_tracks_extra_out_of_src_inputs() {
        let root = tmp_rs_root();
        let crate_dir = root.join("crates").join("sts2-py");
        let asset = root.join("data/some_future_asset.bin");
        let extra = [("data/some_future_asset.bin", asset.clone())];

        let absent = hash_crate(&crate_dir, &extra);
        fs::create_dir_all(root.join("data")).unwrap();
        fs::write(&asset, b"v1").unwrap();
        let v1 = hash_crate(&crate_dir, &extra);
        fs::write(&asset, b"v2").unwrap();
        let v2 = hash_crate(&crate_dir, &extra);

        assert_ne!(absent, v1, "adding a hashed input must change the hash");
        assert_ne!(v1, v2, "editing a hashed input must change the hash");
    }

    #[test]
    fn hash_bytes_deterministic_and_sensitive() {
        assert_eq!(hash_bytes(b"abc"), hash_bytes(b"abc"));
        assert_ne!(hash_bytes(b"abc"), hash_bytes(b"abd"));
        assert_eq!(hash_bytes(b"abc").len(), 16);
    }

    #[test]
    fn hash_crate_is_deterministic() {
        let d = tmp_crate();
        fs::write(d.join("src/lib.rs"), b"fn a() {}").unwrap();
        assert_eq!(hash_crate(&d, &[]), hash_crate(&d, &[]));
    }

    #[test]
    fn hash_crate_changes_on_src_edit() {
        let d = tmp_crate();
        fs::write(d.join("src/lib.rs"), b"fn a() {}").unwrap();
        let h1 = hash_crate(&d, &[]);
        fs::write(d.join("src/lib.rs"), b"fn a() { /* edit */ }").unwrap();
        assert_ne!(
            h1,
            hash_crate(&d, &[]),
            "a src edit must change the hash (Trap 2)"
        );
    }

    #[test]
    fn hash_crate_changes_on_new_src_file() {
        let d = tmp_crate();
        fs::write(d.join("src/lib.rs"), b"fn a() {}").unwrap();
        let h1 = hash_crate(&d, &[]);
        fs::write(d.join("src/extra.rs"), b"fn b() {}").unwrap();
        assert_ne!(
            h1,
            hash_crate(&d, &[]),
            "a new src file must change the hash"
        );
    }

    #[test]
    fn hash_crate_changes_on_nested_src_file() {
        let d = tmp_crate();
        fs::write(d.join("src/lib.rs"), b"fn a() {}").unwrap();
        let h1 = hash_crate(&d, &[]);
        fs::create_dir_all(d.join("src/sub")).unwrap();
        fs::write(d.join("src/sub/mod.rs"), b"fn c() {}").unwrap();
        assert_ne!(
            h1,
            hash_crate(&d, &[]),
            "recursion: a nested src file must change the hash"
        );
    }

    #[test]
    fn hash_crate_changes_on_cargo_toml_edit() {
        let d = tmp_crate();
        fs::write(d.join("src/lib.rs"), b"fn a() {}").unwrap();
        let h1 = hash_crate(&d, &[]);
        fs::write(
            d.join("Cargo.toml"),
            b"[package]\nname=\"x\"\nversion=\"9\"\n",
        )
        .unwrap();
        assert_ne!(
            h1,
            hash_crate(&d, &[]),
            "a Cargo.toml change must change the hash"
        );
    }

    #[test]
    fn hash_crate_extra_is_content_addressed_not_path_addressed() {
        // Two crate dirs with identical src + Cargo.toml + an `extra` of the
        // same LABEL and content but a DIFFERENT absolute path must hash equal —
        // this is what makes the build-time stamp == runtime recompute across a
        // canonical checkout vs a worktree.
        let d1 = tmp_crate();
        fs::write(d1.join("src/lib.rs"), b"fn a() {}").unwrap();
        let data1 = d1.join("data.bin");
        fs::write(&data1, b"v1").unwrap();

        let d2 = tmp_crate();
        fs::write(d2.join("src/lib.rs"), b"fn a() {}").unwrap();
        let data2 = d2.join("data.bin");
        fs::write(&data2, b"v1").unwrap();

        let label = "data/embed.json";
        assert_eq!(
            hash_crate(&d1, &[(label, data1)]),
            hash_crate(&d2, &[(label, data2.clone())]),
            "same label+content => same hash regardless of absolute path",
        );
        // ...but an extra-file CONTENT change must flip it.
        let h_v1 = hash_crate(&d2, &[(label, data2.clone())]);
        fs::write(&data2, b"v2").unwrap();
        assert_ne!(
            h_v1,
            hash_crate(&d2, &[(label, data2)]),
            "extra-file content change must change the hash"
        );
    }

    #[test]
    fn combine_is_ordered_and_deterministic() {
        assert_eq!(combine(&["aa", "bb"]), combine(&["aa", "bb"]));
        assert_ne!(
            combine(&["aa", "bb"]),
            combine(&["bb", "aa"]),
            "order must matter"
        );
        assert_eq!(combine(&["aa", "bb"]).len(), 16);
    }

    #[test]
    fn a_missing_data_input_is_watched_through_its_nearest_directory() {
        let root = std::env::temp_dir().join(format!("srchash-watch-{}", std::process::id()));
        let dir = root.join("data");
        std::fs::create_dir_all(&dir).expect("mkdir");
        let file = dir.join("nets").join("net.json");
        assert_eq!(rerun_watch_path(&root, &file), dir, "absent: nearest dir");
        std::fs::create_dir_all(file.parent().expect("parent")).expect("mkdir");
        std::fs::write(&file, "{}").expect("write");
        assert_eq!(rerun_watch_path(&root, &file), file, "present: the file");
        let outside = root.join("gone").join("x.json");
        std::fs::remove_dir_all(&root).expect("cleanup");
        assert_eq!(
            rerun_watch_path(&root, &outside),
            root,
            "never above rs_root"
        );
    }
}
