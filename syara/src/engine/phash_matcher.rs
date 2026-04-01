//! Perceptual hash matching for binary files (images, audio, video).
//!
//! All matchers produce a 64-bit dHash-style fingerprint. Similarity is
//! `1.0 - hamming_distance / 64.0`. Three built-in implementations:
//!
//! - [`ImageHashMatcher`] — dHash on images via the `image` crate
//! - [`AudioHashMatcher`] — dHash on PCM WAV files (pure Rust WAV reader)
//! - [`VideoHashMatcher`] — byte-sampling fingerprint (no external deps)

use std::path::Path;

use crate::error::SyaraError;
use crate::models::{MatchDetail, PHashRule};

// ── Trait ─────────────────────────────────────────────────────────────────────

/// Perceptual hash matcher.
///
/// Implementations compute a 64-bit hash of a file's content. The default
/// [`match_rule`] compares the scanned file against the reference path stored
/// in the rule.
pub trait PHashMatcher: Send + Sync {
    /// Compute a 64-bit perceptual hash for `file_path`.
    fn compute_hash(&self, file_path: &Path) -> Result<u64, SyaraError>;

    /// Hamming distance between two hashes (number of differing bits).
    fn hamming_distance(&self, hash1: u64, hash2: u64) -> u32 {
        (hash1 ^ hash2).count_ones()
    }

    /// Compare `file_path` (scanned file) against `rule.file_path` (reference).
    ///
    /// Returns a [`MatchDetail`] when `similarity >= rule.threshold`.
    fn match_rule(
        &self,
        rule: &PHashRule,
        file_path: &Path,
    ) -> Result<Vec<MatchDetail>, SyaraError> {
        let ref_path = Path::new(&rule.file_path);
        if !ref_path.exists() {
            return Err(SyaraError::FileNotFound(rule.file_path.clone()));
        }
        let ref_hash = self.compute_hash(ref_path)?;
        let input_hash = self.compute_hash(file_path)?;
        let distance = self.hamming_distance(ref_hash, input_hash);
        let similarity = 1.0 - (distance as f64 / 64.0);
        if similarity >= rule.threshold {
            let mut detail = MatchDetail::new(
                rule.identifier.clone(),
                file_path.to_string_lossy().into_owned(),
            );
            detail.score = similarity;
            detail.explanation = format!("PHash similarity: {similarity:.3}");
            Ok(vec![detail])
        } else {
            Ok(vec![])
        }
    }
}

// ── Image matcher ─────────────────────────────────────────────────────────────

/// dHash matcher for image files.
///
/// Converts to grayscale, resizes to `(hash_size + 1) × hash_size`, and
/// encodes the horizontal gradient as a 64-bit hash. Default `hash_size = 8`
/// produces a 64-bit hash.
pub struct ImageHashMatcher {
    hash_size: u32,
}

impl ImageHashMatcher {
    pub fn new(hash_size: u32) -> Self {
        Self { hash_size }
    }
}

impl Default for ImageHashMatcher {
    fn default() -> Self {
        Self::new(8)
    }
}

impl PHashMatcher for ImageHashMatcher {
    fn compute_hash(&self, file_path: &Path) -> Result<u64, SyaraError> {
        let img = image::open(file_path)
            .map_err(|e| SyaraError::PhashError(e.to_string()))?
            .into_luma8();

        let width = self.hash_size + 1;
        let height = self.hash_size;

        let img =
            image::imageops::resize(&img, width, height, image::imageops::FilterType::Lanczos3);

        let mut hash: u64 = 0;
        for row in 0..height {
            for col in 0..self.hash_size {
                let left = img.get_pixel(col, row).0[0];
                let right = img.get_pixel(col + 1, row).0[0];
                if left > right {
                    hash |= 1u64 << (row * self.hash_size + col);
                }
            }
        }
        Ok(hash)
    }
}

// ── Audio (WAV) matcher ───────────────────────────────────────────────────────

/// dHash-style matcher for PCM WAV audio files.
///
/// Samples 65 evenly-spaced frames from the `data` chunk, then builds a
/// 64-bit hash by comparing consecutive amplitude values. Uses a minimal
/// pure-Rust WAV reader — no external audio crate required.
pub struct AudioHashMatcher;

impl Default for AudioHashMatcher {
    fn default() -> Self {
        Self
    }
}

impl PHashMatcher for AudioHashMatcher {
    fn compute_hash(&self, file_path: &Path) -> Result<u64, SyaraError> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let mut f = File::open(file_path).map_err(|e| SyaraError::PhashError(e.to_string()))?;

        // Validate RIFF/WAVE header
        let mut header = [0u8; 12];
        f.read_exact(&mut header)
            .map_err(|e| SyaraError::PhashError(e.to_string()))?;
        if &header[0..4] != b"RIFF" || &header[8..12] != b"WAVE" {
            return Err(SyaraError::PhashError("not a valid WAV file".into()));
        }

        // Scan chunks for fmt + data
        let mut sample_width: u64 = 2; // bytes per sample (default 16-bit)
        let mut data_offset: u64 = 0;
        let mut n_frames: u64 = 0;

        loop {
            let mut id = [0u8; 4];
            let mut sz = [0u8; 4];
            if f.read_exact(&mut id).is_err() || f.read_exact(&mut sz).is_err() {
                break;
            }
            let chunk_len = u32::from_le_bytes(sz) as u64;

            if &id == b"fmt " {
                let mut fmt = vec![0u8; chunk_len as usize];
                f.read_exact(&mut fmt)
                    .map_err(|e| SyaraError::PhashError(e.to_string()))?;
                // bits_per_sample at offset 14 in fmt chunk body
                if fmt.len() >= 16 {
                    let bits = u16::from_le_bytes([fmt[14], fmt[15]]);
                    sample_width = (bits / 8).max(1) as u64;
                }
            } else if &id == b"data" {
                data_offset = f
                    .stream_position()
                    .map_err(|e| SyaraError::PhashError(e.to_string()))?;
                n_frames = chunk_len / sample_width.max(1);
                break;
            } else {
                f.seek(SeekFrom::Current(chunk_len as i64))
                    .map_err(|e| SyaraError::PhashError(e.to_string()))?;
            }
        }

        if n_frames == 0 {
            return Ok(0);
        }

        // Sample 65 evenly-spaced frames
        let n_samples: u64 = 65;
        let step = (n_frames / n_samples).max(1);
        let mut samples: Vec<i32> = Vec::with_capacity(n_samples as usize);

        for i in 0..n_samples {
            let pos = (i * step).min(n_frames - 1);
            f.seek(SeekFrom::Start(data_offset + pos * sample_width))
                .map_err(|e| SyaraError::PhashError(e.to_string()))?;
            let mut raw = vec![0u8; sample_width as usize];
            let val = if f.read_exact(&mut raw).is_ok() {
                match sample_width {
                    1 => (raw[0] as i32) - 128,
                    2 => i16::from_le_bytes([raw[0], raw[1]]) as i32,
                    4 => i32::from_le_bytes([raw[0], raw[1], raw[2], raw[3]]),
                    _ => raw[0] as i32,
                }
            } else {
                0
            };
            samples.push(val);
        }

        // Build 64-bit hash: bit i set when sample[i] > sample[i+1]
        let mut hash: u64 = 0;
        for i in 0..64usize {
            if i + 1 < samples.len() && samples[i] > samples[i + 1] {
                hash |= 1u64 << i;
            }
        }
        Ok(hash)
    }
}

// ── Video (raw bytes) matcher ─────────────────────────────────────────────────

/// Byte-sampling fingerprint for video (or any binary) files.
///
/// Reads 65 bytes at evenly-spaced positions across the file and encodes each
/// consecutive pair comparison as a bit. Requires no external dependencies.
pub struct VideoHashMatcher;

impl Default for VideoHashMatcher {
    fn default() -> Self {
        Self
    }
}

impl PHashMatcher for VideoHashMatcher {
    fn compute_hash(&self, file_path: &Path) -> Result<u64, SyaraError> {
        use std::fs::File;
        use std::io::{Read, Seek, SeekFrom};

        let file_size = file_path
            .metadata()
            .map_err(|e| SyaraError::PhashError(e.to_string()))?
            .len();

        if file_size == 0 {
            return Ok(0);
        }

        let mut f = File::open(file_path).map_err(|e| SyaraError::PhashError(e.to_string()))?;
        let n_samples: u64 = 65;
        let mut samples = Vec::with_capacity(n_samples as usize);

        for i in 0..n_samples {
            let pos = if file_size > 1 {
                i * (file_size - 1) / (n_samples - 1)
            } else {
                0
            };
            f.seek(SeekFrom::Start(pos))
                .map_err(|e| SyaraError::PhashError(e.to_string()))?;
            let mut byte = [0u8; 1];
            samples.push(if f.read_exact(&mut byte).is_ok() { byte[0] } else { 0 });
        }

        let mut hash: u64 = 0;
        for i in 0..64usize {
            if samples[i] > samples[i + 1] {
                hash |= 1u64 << i;
            }
        }
        Ok(hash)
    }
}

// ── Tests ─────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    // ── Trait unit tests ──────────────────────────────────────────────────────

    struct FixedHashMatcher(u64);

    impl PHashMatcher for FixedHashMatcher {
        fn compute_hash(&self, _: &Path) -> Result<u64, SyaraError> {
            Ok(self.0)
        }
    }

    #[test]
    fn hamming_distance_identical() {
        let m = FixedHashMatcher(0);
        assert_eq!(m.hamming_distance(0xDEADBEEF, 0xDEADBEEF), 0);
    }

    #[test]
    fn hamming_distance_known() {
        let m = FixedHashMatcher(0);
        // 0b0000 vs 0b1111 → 4 differing bits
        assert_eq!(m.hamming_distance(0b0000u64, 0b1111u64), 4);
    }

    #[test]
    fn match_rule_above_threshold() {
        // Two identical hashes → similarity = 1.0 ≥ 0.9
        let m = FixedHashMatcher(0xABCD);

        // Write two identical temp files (content irrelevant; matcher ignores it)
        let ref_file = temp_file(b"ref");
        let input_file = temp_file(b"input");

        let rule = PHashRule {
            identifier: "$ph1".into(),
            file_path: ref_file.path().to_string_lossy().into_owned(),
            threshold: 0.9,
            phash_name: "imagehash".into(),
        };

        let results = m.match_rule(&rule, input_file.path()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].identifier, "$ph1");
        assert!((results[0].score - 1.0).abs() < 1e-9);
        assert!(results[0].explanation.contains("PHash similarity:"));
    }

    #[test]
    fn match_rule_below_threshold() {
        // We need a matcher that returns 0 for ref and MAX for input —
        struct PairMatcher { call: std::sync::atomic::AtomicU64 }
        impl PHashMatcher for PairMatcher {
            fn compute_hash(&self, _: &Path) -> Result<u64, SyaraError> {
                let n = self.call.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                Ok(if n == 0 { 0u64 } else { u64::MAX })
            }
        }

        let ref_file = temp_file(b"ref");
        let input_file = temp_file(b"input");

        let rule = PHashRule {
            identifier: "$ph2".into(),
            file_path: ref_file.path().to_string_lossy().into_owned(),
            threshold: 0.9,
            phash_name: "imagehash".into(),
        };

        let m = PairMatcher { call: std::sync::atomic::AtomicU64::new(0) };
        let results = m.match_rule(&rule, input_file.path()).unwrap();
        assert!(results.is_empty());
    }

    // ── VideoHashMatcher tests ────────────────────────────────────────────────

    #[test]
    fn video_hash_empty_file_returns_zero() {
        let f = temp_file(b"");
        let hash = VideoHashMatcher.compute_hash(f.path()).unwrap();
        assert_eq!(hash, 0);
    }

    #[test]
    fn video_hash_deterministic() {
        let data = b"the quick brown fox jumps over the lazy dog 0123456789abcdef";
        let f = temp_file(data);
        let h1 = VideoHashMatcher.compute_hash(f.path()).unwrap();
        let h2 = VideoHashMatcher.compute_hash(f.path()).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn video_hash_differs_for_different_content() {
        // Ascending bytes → each sample[i] < sample[i+1] → hash ≈ 0
        // Descending bytes → each sample[i] > sample[i+1] → hash ≈ MAX
        let asc: Vec<u8> = (0u8..=127).collect();
        let desc: Vec<u8> = (0u8..=127).rev().collect();
        let f1 = temp_file(&asc);
        let f2 = temp_file(&desc);
        let h1 = VideoHashMatcher.compute_hash(f1.path()).unwrap();
        let h2 = VideoHashMatcher.compute_hash(f2.path()).unwrap();
        assert_ne!(h1, h2);
    }

    // ── AudioHashMatcher tests ────────────────────────────────────────────────

    #[test]
    fn audio_hash_invalid_file_returns_error() {
        let f = temp_file(b"not a wav file at all");
        let result = AudioHashMatcher.compute_hash(f.path());
        assert!(result.is_err());
    }

    #[test]
    fn audio_hash_deterministic() {
        let wav = minimal_wav(44100, &sawtooth_samples(65));
        let f = temp_file(&wav);
        let h1 = AudioHashMatcher.compute_hash(f.path()).unwrap();
        let h2 = AudioHashMatcher.compute_hash(f.path()).unwrap();
        assert_eq!(h1, h2);
    }

    #[test]
    fn audio_hash_empty_data_returns_zero() {
        let wav = minimal_wav(44100, &[]);
        let f = temp_file(&wav);
        let hash = AudioHashMatcher.compute_hash(f.path()).unwrap();
        assert_eq!(hash, 0);
    }

    // ── Helpers ───────────────────────────────────────────────────────────────

    /// Write `data` to a named temporary file; returns the guard.
    fn temp_file(data: &[u8]) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(data).unwrap();
        f
    }

    /// Build a minimal valid PCM WAV byte vector with 16-bit mono samples.
    fn minimal_wav(sample_rate: u32, samples: &[i16]) -> Vec<u8> {
        let data_len = (samples.len() * 2) as u32;
        let file_len = 36 + data_len;

        let mut v: Vec<u8> = Vec::with_capacity(file_len as usize + 8);
        // RIFF header
        v.extend_from_slice(b"RIFF");
        v.extend_from_slice(&file_len.to_le_bytes());
        v.extend_from_slice(b"WAVE");
        // fmt chunk (16 bytes body)
        v.extend_from_slice(b"fmt ");
        v.extend_from_slice(&16u32.to_le_bytes());
        v.extend_from_slice(&1u16.to_le_bytes()); // PCM
        v.extend_from_slice(&1u16.to_le_bytes()); // mono
        v.extend_from_slice(&sample_rate.to_le_bytes());
        v.extend_from_slice(&(sample_rate * 2).to_le_bytes()); // byte rate
        v.extend_from_slice(&2u16.to_le_bytes()); // block align
        v.extend_from_slice(&16u16.to_le_bytes()); // bits per sample
        // data chunk
        v.extend_from_slice(b"data");
        v.extend_from_slice(&data_len.to_le_bytes());
        for &s in samples {
            v.extend_from_slice(&s.to_le_bytes());
        }
        v
    }

    /// Generate a simple sawtooth wave of `n` 16-bit samples.
    fn sawtooth_samples(n: usize) -> Vec<i16> {
        (0..n).map(|i| ((i as i32 * 1000) % i16::MAX as i32) as i16).collect()
    }
}
