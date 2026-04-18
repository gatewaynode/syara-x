//! Integration tests for the `syara-x-capi` C FFI.
//!
//! Two surfaces:
//! 1. `header_matches_regenerated_cbindgen_output` (default) — regenerates
//!    `syara_x.h` in-process using the same cbindgen configuration as
//!    `build.rs` and diffs the result against the checked-in
//!    `capi/syara_x.h`. Catches the "edited the header by hand, diverged
//!    from src/lib.rs" bug class. The build-script regenerates the header
//!    on every `cargo build`, so this test is tight: if the on-disk header
//!    differs from what cbindgen produces for the current source, the
//!    source or header was tampered with outside the normal build cycle.
//! 2. `integration_real_c_link` (`#[ignore]`, Unix-only) — compiles the
//!    capi staticlib via `cargo build -p syara-x-capi --release`, writes
//!    a small C driver to a tempdir that `#include`s `syara_x.h`, invokes
//!    the system `cc` to link it against `libsyara_x_capi.a`, runs the
//!    binary, and asserts that `HIT:r` appears in stdout. Exercises the
//!    full ABI round-trip from C through the staticlib back to the C caller.
//!
//! The `#[ignore]` test synthesizes its C source at runtime so the repo
//! does not grow a committed `.c` fixture that would need its own
//! maintenance.

use std::path::Path;

const CRATE_DIR: &str = env!("CARGO_MANIFEST_DIR");

/// Regenerate the header in-process using the same cbindgen builder
/// settings as `capi/build.rs`. Any divergence here would invalidate the
/// drift test below.
fn regenerate_header() -> String {
    let tmp = tempfile::NamedTempFile::new().expect("tempfile for regen");
    cbindgen::Builder::new()
        .with_crate(CRATE_DIR)
        .with_language(cbindgen::Language::C)
        .with_include_guard("SYARA_X_H")
        .with_documentation(true)
        .with_tab_width(4)
        .generate()
        .expect("cbindgen regeneration failed")
        .write_to_file(tmp.path());
    std::fs::read_to_string(tmp.path()).expect("read regenerated header")
}

fn normalize(s: &str) -> String {
    s.lines()
        .map(str::trim_end)
        .collect::<Vec<_>>()
        .join("\n")
}

/// The checked-in `capi/syara_x.h` must match what cbindgen produces for
/// the current `capi/src/lib.rs`. On mismatch, `cargo build -p syara-x-capi`
/// regenerates it.
#[test]
fn header_matches_regenerated_cbindgen_output() {
    let header_path = Path::new(CRATE_DIR).join("syara_x.h");
    let checked_in = std::fs::read_to_string(&header_path).expect("read syara_x.h");
    let regenerated = regenerate_header();

    let a = normalize(&checked_in);
    let b = normalize(&regenerated);
    assert_eq!(
        a, b,
        "{} is out of sync with capi/src/lib.rs. \
         Run `cargo build -p syara-x-capi` to regenerate.",
        header_path.display()
    );
}

// ── Real-backend test (opt-in, invokes the system `cc`) ────────────────────

/// Run with:
///   cargo test -p syara-x-capi -- --ignored --nocapture integration_real_c_link
///
/// Requires a working `cc` on PATH. macOS (Apple `cc`) and Linux (gcc/clang)
/// are both supported. Other targets are skipped at compile time via
/// `#[cfg(unix)]`.
#[test]
#[ignore]
#[cfg(unix)]
fn integration_real_c_link() {
    use std::process::Command;

    let workspace_root = Path::new(CRATE_DIR)
        .parent()
        .expect("capi has workspace parent");

    let cargo = std::env::var("CARGO").unwrap_or_else(|_| "cargo".to_string());
    let build_status = Command::new(&cargo)
        .args(["build", "-p", "syara-x-capi", "--release"])
        .current_dir(workspace_root)
        .status()
        .expect("failed to invoke cargo");
    assert!(build_status.success(), "cargo build -p syara-x-capi --release failed");

    let staticlib = workspace_root.join("target/release/libsyara_x_capi.a");
    assert!(
        staticlib.exists(),
        "expected staticlib at {} — did the build complete?",
        staticlib.display()
    );

    let tmpdir = tempfile::tempdir().expect("tempdir");
    let c_src_path = tmpdir.path().join("driver.c");
    let bin_path = tmpdir.path().join("driver");

    const C_SRC: &str = r#"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "syara_x.h"

int main(void) {
    SyaraRules *rules = NULL;
    enum SyaraStatus st = syara_compile_str(
        "rule r { strings: $s = \"evil\" condition: $s }", &rules);
    if (st != SyaraOk || rules == NULL) {
        fprintf(stderr, "compile failed: %d\n", (int)st);
        return 1;
    }

    SyaraMatchArray *matches = NULL;
    st = syara_scan(rules, "the evil word", &matches);
    if (st != SyaraOk || matches == NULL) {
        fprintf(stderr, "scan failed: %d\n", (int)st);
        syara_rules_free(rules);
        return 2;
    }

    for (size_t i = 0; i < matches->count; i++) {
        const char *name = matches->matches[i].rule_name;
        if (matches->matches[i].matched) {
            printf("HIT:%s\n", name);
        } else {
            printf("MISS:%s\n", name);
        }
    }

    syara_matches_free(matches);
    syara_rules_free(rules);
    return 0;
}
"#;
    std::fs::write(&c_src_path, C_SRC).expect("write driver.c");

    let cc = std::env::var("CC").unwrap_or_else(|_| "cc".to_string());
    let mut cmd = Command::new(&cc);
    cmd.arg("-I")
        .arg(CRATE_DIR)
        .arg("-o")
        .arg(&bin_path)
        .arg(&c_src_path)
        .arg(&staticlib);

    #[cfg(target_os = "macos")]
    cmd.args(["-framework", "CoreFoundation", "-framework", "Security"]);
    #[cfg(target_os = "linux")]
    cmd.args(["-lpthread", "-ldl", "-lm"]);

    let cc_output = cmd.output().expect("invoke cc");
    assert!(
        cc_output.status.success(),
        "cc failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&cc_output.stdout),
        String::from_utf8_lossy(&cc_output.stderr)
    );

    let run_output = Command::new(&bin_path).output().expect("run driver");
    assert!(
        run_output.status.success(),
        "driver failed:\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&run_output.stdout),
        String::from_utf8_lossy(&run_output.stderr)
    );

    let stdout = String::from_utf8_lossy(&run_output.stdout);
    assert!(
        stdout.contains("HIT:r"),
        "expected 'HIT:r' in driver stdout, got:\n{}",
        stdout
    );
}
