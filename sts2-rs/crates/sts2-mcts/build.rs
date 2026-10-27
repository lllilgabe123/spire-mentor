//! Three jobs:
//!
//! - Stamp a content hash of this crate's build inputs into
//!   `STS2_MCTS_SRC_HASH`, exposed as `sts2_mcts::SRC_HASH`. This is part of
//!   the content-addressed engine-identity fix (issue #15, bead
//!   `sts2-rs-cg5ci`); see sts2-core/build.rs for the rationale. The inputs
//!   are `src`, `Cargo.toml`, the built-in playout nets and the built-in move
//!   prior (`sts2_srchash::MCTS_DATA_INPUTS`), which live outside the package.
//!   So the stamp names its inputs explicitly instead of relying on cargo's
//!   any-file-in-the-package default.
//! - Set `cfg(rollout_net_<character>)` for each built-in net file present
//!   (`data/rollout_nets/<character>.json`). `src/rollout_net.rs` compiles a
//!   net in only then, so a checkout without the file still compiles and that
//!   character plays the heuristic playout.
//! - Set `cfg(prior_net_<character>)` for each built-in move-prior file
//!   present (`data/prior_nets/<character>.json`, GH #830; Ironclad only).
//!   `rollout_net::builtin_prior_json` compiles the text in only then. A
//!   checkout without it still compiles, and the test
//!   `the_ironclad_prior_net_is_compiled_in` fails instead of letting that
//!   build search without its prior unnoticed.

const NETS: [&str; 5] = ["ironclad", "silent", "defect", "regent", "necrobinder"];

/// The characters whose move prior may be compiled in. Dropping a file for
/// another character into `data/prior_nets/` must not switch it on.
const PRIOR_NETS: [&str; 1] = ["ironclad"];

fn main() {
    sts2_srchash::emit_build_stamp("sts2-mcts", "STS2_MCTS_SRC_HASH");

    let crate_dir = std::path::PathBuf::from(
        std::env::var_os("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR set by cargo"),
    );
    for name in NETS {
        println!("cargo::rustc-check-cfg=cfg(rollout_net_{name})");
        // `emit_build_stamp` already watches this path (or, while it is
        // absent, its nearest existing directory), so adding or removing the
        // file flips the cfg without a `cargo clean`.
        let net = crate_dir.join(format!("../../data/rollout_nets/{name}.json"));
        if net.is_file() {
            println!("cargo:rustc-cfg=rollout_net_{name}");
        }
    }
    for name in PRIOR_NETS {
        println!("cargo::rustc-check-cfg=cfg(prior_net_{name})");
        // Watched by `emit_build_stamp` too: each file is an
        // `MCTS_DATA_INPUTS` entry.
        let net = crate_dir.join(format!("../../data/prior_nets/{name}.json"));
        if net.is_file() {
            println!("cargo:rustc-cfg=prior_net_{name}");
        }
    }
}
