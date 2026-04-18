//! Integration tests for `phash:` rules end-to-end.
//!
//! Three surfaces:
//! 1. Deterministic: `FixedHashMatcher` → parse → `scan_file()` (runs by default).
//! 2. `#[ignore] integration_real_image_phash` — generates a PNG via the
//!    `image` crate at runtime and runs the registered `imagehash` matcher.
//! 3. `#[ignore] integration_real_wav_phash` — generates a PCM WAV at
//!    runtime and runs the registered `audiohash` matcher.
//!
//! The `#[ignore]` tests synthesize their inputs in a temp dir so the repo
//! does not grow committed binary fixtures.

#![cfg(feature = "phash")]

use std::collections::HashMap;
use std::io::Write;
use std::path::{Path, PathBuf};

use syara_x::compile_str;
use syara_x::engine::phash_matcher::PHashMatcher;
use syara_x::error::SyaraError;

/// Hash matcher backed by a path → 64-bit hash lookup table.
///
/// Unknown paths return `u64::MAX` so any reference-vs-input pair where one
/// side is unknown lands at maximum hamming distance (similarity 0.0).
struct FixedHashMatcher {
    table: HashMap<PathBuf, u64>,
}

impl PHashMatcher for FixedHashMatcher {
    fn compute_hash(&self, file_path: &Path) -> Result<u64, SyaraError> {
        Ok(self.table.get(file_path).copied().unwrap_or(u64::MAX))
    }
}

/// End-to-end: parse → compile → register custom matcher → `scan_file`.
///
/// Verifies the cost-ordered engine wires phash through correctly:
/// matching pair scores 1.0, mismatched pair scores 0.0 (64-bit hamming).
#[test]
fn phash_scan_deterministic_match_and_miss() {
    let ref_file = tempfile::NamedTempFile::new().expect("ref tempfile");
    let match_file = tempfile::NamedTempFile::new().expect("match tempfile");
    let miss_file = tempfile::NamedTempFile::new().expect("miss tempfile");

    let ref_path = ref_file.path().to_path_buf();
    let match_path = match_file.path().to_path_buf();
    let miss_path = miss_file.path().to_path_buf();

    // ref + match → hash 0; miss → hash u64::MAX (64 differing bits).
    let mut table: HashMap<PathBuf, u64> = HashMap::new();
    table.insert(ref_path.clone(), 0);
    table.insert(match_path.clone(), 0);
    table.insert(miss_path.clone(), u64::MAX);

    let src = format!(
        r#"
rule phash_match: security
{{
    phash:
        $p = "{}" threshold=0.9 hasher="fixed-phash"

    condition:
        $p
}}
"#,
        ref_path.display()
    );

    let mut rules = compile_str(&src).expect("compile");
    rules.register_phash_matcher(
        "fixed-phash",
        Box::new(FixedHashMatcher {
            table: table.clone(),
        }),
    );

    // Positive: identical hashes → similarity 1.0 ≥ 0.9
    let results = rules.scan_file(&match_path);
    let hit = results
        .iter()
        .find(|m| m.rule_name == "phash_match")
        .expect("rule present in results");
    assert!(hit.matched, "identical hash → similarity 1.0 ≥ 0.9");
    let details = hit
        .matched_patterns
        .get("$p")
        .expect("$p populated on positive match");
    assert_eq!(details.len(), 1);
    assert_eq!(details[0].identifier, "$p");
    assert!(
        (details[0].score - 1.0).abs() < 1e-9,
        "expected score 1.0, got {}",
        details[0].score
    );
    assert!(
        details[0].explanation.contains("PHash similarity:"),
        "explanation should describe phash similarity"
    );

    // Negative: 64 differing bits → similarity 0.0 < 0.9
    let results = rules.scan_file(&miss_path);
    let hit = results
        .iter()
        .find(|m| m.rule_name == "phash_match")
        .expect("rule present in results");
    assert!(!hit.matched, "u64::MAX vs 0 → similarity 0.0 < 0.9");
}

// ── Real-backend tests (opt-in, synthesize inputs at runtime) ──────────────

/// Run with:
///   cargo test -p syara-x --features phash -- --ignored --nocapture integration_real_image_phash
#[test]
#[ignore]
fn integration_real_image_phash() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path_a = dir.path().join("gradient_a.png");
    let path_b = dir.path().join("gradient_b.png");
    let path_c = dir.path().join("gradient_inverted.png");

    // Identical content → identical dHash → similarity 1.0
    write_gradient_png(&path_a, false);
    write_gradient_png(&path_b, false);
    // Inverted gradient → many bits flip in dHash → similarity drops
    write_gradient_png(&path_c, true);

    let src = format!(
        r#"
rule image_phash {{
    phash:
        $p = "{}" threshold=0.95 hasher="imagehash"
    condition:
        $p
}}
"#,
        path_a.display()
    );

    let rules = compile_str(&src).expect("compile");

    let results = rules.scan_file(&path_b);
    let hit = results
        .iter()
        .find(|m| m.rule_name == "image_phash")
        .expect("rule present");
    assert!(hit.matched, "identical PNG content should match at 0.95");

    let results = rules.scan_file(&path_c);
    let hit = results
        .iter()
        .find(|m| m.rule_name == "image_phash")
        .expect("rule present");
    assert!(
        !hit.matched,
        "inverted-gradient PNG should not match at 0.95"
    );
}

/// Run with:
///   cargo test -p syara-x --features phash -- --ignored --nocapture integration_real_wav_phash
#[test]
#[ignore]
fn integration_real_wav_phash() {
    let dir = tempfile::tempdir().expect("tempdir");
    let ref_path = dir.path().join("ref.wav");
    let match_path = dir.path().join("match.wav");
    write_sawtooth_wav(&ref_path);
    write_sawtooth_wav(&match_path);

    let src = format!(
        r#"
rule wav_phash {{
    phash:
        $p = "{}" threshold=0.95 hasher="audiohash"
    condition:
        $p
}}
"#,
        ref_path.display()
    );

    let rules = compile_str(&src).expect("compile");
    let results = rules.scan_file(&match_path);
    let hit = results
        .iter()
        .find(|m| m.rule_name == "wav_phash")
        .expect("rule present");
    assert!(hit.matched, "identical WAV → similarity 1.0 ≥ 0.95");
}

// ── Helpers ───────────────────────────────────────────────────────────────

fn write_gradient_png(path: &Path, reverse: bool) {
    let img = image::GrayImage::from_fn(32, 32, |x, _y| {
        let v = (x * 255 / 31) as u8;
        image::Luma([if reverse { 255 - v } else { v }])
    });
    img.save(path).expect("write png");
}

fn write_sawtooth_wav(path: &Path) {
    let samples: Vec<i16> = (0..128)
        .map(|i| ((i as i32 * 1000) % i16::MAX as i32) as i16)
        .collect();
    let sample_rate: u32 = 44100;
    let channels: u16 = 1;
    let block_align: u16 = channels * 2;
    let byte_rate: u32 = sample_rate * block_align as u32;
    let data_len = (samples.len() * 2) as u32;
    let file_len = 36 + data_len;

    let mut v: Vec<u8> = Vec::with_capacity(file_len as usize + 8);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&file_len.to_le_bytes());
    v.extend_from_slice(b"WAVE");
    v.extend_from_slice(b"fmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&channels.to_le_bytes());
    v.extend_from_slice(&sample_rate.to_le_bytes());
    v.extend_from_slice(&byte_rate.to_le_bytes());
    v.extend_from_slice(&block_align.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for &s in &samples {
        v.extend_from_slice(&s.to_le_bytes());
    }

    let mut f = std::fs::File::create(path).expect("create wav");
    f.write_all(&v).expect("write wav");
}
