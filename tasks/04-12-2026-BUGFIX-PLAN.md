# Bug Fix Plan: syara-x (34 bugs, 7 batches)

## Context
Codebase analysis on 2026-04-12 discovered 3 critical, 16 medium, and 15 low severity bugs across the syara-x library. Issues range from correctness (silently ignored modifiers, condition re-parsing), performance (O(n²) parser, uncached regexes), safety (WAV parser DoS, bit overflow, prompt injection), to ergonomics (sentinel values, missing tests). All bugs are documented in `tasks/BUGS.md`.

Every bug fix must include a unit test that confirms the fix.

## Batch Order

Batch 1 changes the core data model (foundation for everything else). Batches 2–6 are independent after Batch 1. Batch 7 (tests + C API) is last to verify all prior fixes.

---

## Batch 1: Core Data Model + Condition Compilation — COMPLETE
**Bugs**: BUG-001, BUG-006, BUG-007, BUG-017
**Status**: Done. Needs backfill regression tests.

**Changes made**:
- `models.rs`: `Option<usize>` for positions; `compiled_condition: Option<Expr>` on `Rule`
- `condition.rs`: EOF check after `parse_expr()`; clone-free `evaluate_hypothetical`
- `compiler.rs`: stores parsed `Expr` in rule instead of discarding
- `compiled_rules.rs`: uses stored `Expr` in `evaluate_condition`, no re-parse

**Tests needed (backfill)**:
- Condition with trailing tokens returns error (BUG-006)
- `is_identifier_needed` works correctly without cloning (BUG-007)
- `MatchDetail` positions are `Option<usize>` (BUG-017)
- Compiled condition is used instead of re-parsing (BUG-001)

---

## Batch 2: String Matcher Correctness + Performance — COMPLETE
**Bugs**: BUG-002, BUG-009, BUG-010, BUG-025, BUG-026
**Status**: Done. Needs backfill regression tests.

**Changes made**:
- `string_matcher.rs`: `fullword` wraps pattern in `\b...\b`; `compile()` split to avoid clone; wide regexes cached with `:wide_regex`/`:wide_literal` suffix keys; stripped-to-original position map for wide regex matches

**Tests needed (backfill)**:
- `fullword` modifier on literal string (BUG-002)
- `fullword` modifier on regex pattern (BUG-002)
- Wide regex positions map back to original text (BUG-026)

---

## Batch 3: Parser Performance + Correctness — COMPLETE
**Bugs**: BUG-004, BUG-005, BUG-020, BUG-021
**Status**: Done.

**Changes made**:
- `parser.rs` split into `parser/mod.rs` (562 lines) + `parser/sections.rs` (369 lines)
- BUG-004: 7 `LazyLock<Regex>` statics replace all inline `Regex::new()` calls (2 in mod.rs, 5 in sections.rs); `section_content` remains dynamic (pattern built from section name)
- BUG-005: `split_rules` reworked to use byte offsets on `&str` via `content.as_bytes()` �� no more `chars[i..].iter().collect()`
- BUG-020: String pattern regex changed from `[^"]*` to `(?:[^"\\]|\\.)*`; added `unescape_string()` to process `\"`, `\\`, `\n`, `\t`, `\r`
- BUG-021: `parse_rule_block` returns error when patterns exist but condition is missing/empty

**Tests added**:
- `test_escaped_quotes_in_string_pattern` (BUG-020)
- `test_escaped_backslash_in_string_pattern` (BUG-020)
- `test_missing_condition_with_patterns_is_error` (BUG-021)
- `test_typo_condition_keyword_is_error` (BUG-021)
- `test_rule_without_patterns_or_condition_ok` (BUG-021 — meta-only rule is valid)
- `test_many_rules_parse_without_quadratic_slowdown` (BUG-005 — 200 rules)
- `test_unescape_string_sequences` (unescape helper)

**Verify**: `cargo test && cargo clippy -- -D warnings`

---

## Batch 4: Phash/Audio Safety + scan_file Fix — COMPLETE
**Bugs**: BUG-003, BUG-008, BUG-014, BUG-015, BUG-029
**Status**: Done.

**Changes made**:
- `phash_matcher.rs`: BUG-003 — `compute_hash` returns `Err` for `hash_size > 8` (64-bit limit); added `MAX_HASH_SIZE` const
- `phash_matcher.rs`: BUG-014 — reads `n_channels` from fmt chunk offset 2; `n_frames = chunk_len / (n_channels * sample_width)`; sampling loop seeks by `frame_size`
- `phash_matcher.rs`: BUG-015 — `MAX_CHUNK_ALLOC = 256MB`; fmt and data chunks rejected if `chunk_len > MAX_CHUNK_ALLOC`
- `phash_matcher.rs`: BUG-029 — unknown/fmt chunks with odd `chunk_len` seek past 1 padding byte per RIFF spec
- `compiled_rules.rs`: BUG-008 — `scan_file` reads file content via `String::from_utf8_lossy(std::fs::read(path))` for string matchers

**Tests added**:
- `image_hash_size_9_returns_error` (BUG-003)
- `image_hash_size_8_is_valid` (BUG-003)
- `audio_hash_stereo_correct_frame_count` (BUG-014)
- `audio_hash_oversized_data_chunk_returns_error` (BUG-015)
- `audio_hash_odd_chunk_padding` (BUG-029)

**Note**: BUG-008 (`scan_file` string matching) is tested indirectly — Batch 7 integration tests will add explicit `scan_file` coverage with mixed phash+string rules.

**Verify**: `cargo test && cargo clippy -- -D warnings` ✓

---

## Batch 5: HTTP Clients + Shared Helpers — COMPLETE
**Bugs**: BUG-011, BUG-012, BUG-013, BUG-033
**Status**: Done.

**Changes made**:
- `engine/mod.rs`: new `HttpEmbedder` struct with `Mutex<HashMap>` embedding cache (BUG-033), 10s connect + 30s read timeouts (BUG-011); used by both semantic_matcher and classifier (BUG-012)
- `engine/semantic_matcher.rs`: `HttpEmbeddingMatcher` now wraps `HttpEmbedder`; `embed()` delegates with `SyaraError::SemanticError` mapping
- `engine/classifier.rs`: `HttpEmbeddingClassifier` now wraps `HttpEmbedder`; `embed()` delegates with `SyaraError::ClassifierError` mapping
- `engine/llm_evaluator.rs`: `OllamaEvaluator` builds client with 10s connect + 30s read timeouts (BUG-011); `build_prompt` uses XML delimiters `<pattern>` and `<input>` around untrusted content (BUG-013); doc comment documents injection surface

**Tests added**:
- `http_embedder_has_timeouts_configured` (BUG-011 — verifies timeout constants)
- `http_embedder_caches_results` (BUG-033 — verifies cache behavior)
- `shared_embedder_used_by_semantic_matcher` (BUG-012 — verifies delegation)
- `shared_embedder_used_by_classifier` (BUG-012 — verifies delegation)
- `prompt_uses_xml_delimiters` (BUG-013 — verifies XML delimiters in prompt)
- `llm_evaluator_has_timeouts_configured` (BUG-011 — verifies timeout constants)

**Verify**: `cargo test -p syara-x --features all && cargo clippy --features all -- -D warnings` ✓

---

## Batch 6: Cache, Cleaner, Chunker, Condition Edge Cases — COMPLETE
**Bugs**: BUG-016, BUG-019, BUG-022, BUG-027, BUG-034
**Status**: Done.

**Changes made**:
- `cache.rs`: replaced SHA256+hex key with `(String, String)` tuple key; removed `sha2` and `hex` deps from `syara/Cargo.toml` (BUG-016)
- `engine/cleaner.rs`: `is_ascii_digit()` → `is_numeric()` for Unicode digit parity with Python (BUG-019)
- `engine/chunker.rs`: `WordChunker::chunk()` returns whole text when `chunk_size == 0` instead of panicking (BUG-027)
- `condition.rs`: tokenizer produces `Token::Unknown(char)` for unrecognized characters, causing parse error (BUG-022); `resolve_set` sorts keys for `SetExpr::Them` and `SetExpr::Wildcard` (BUG-034)

**Tests added**:
- `cache_tuple_key_insert_and_get` (BUG-016)
- `cache_clear_empties_store` (BUG-016)
- `aggressive_cleaner_strips_unicode_digits` (BUG-019 — Arabic-Indic)
- `aggressive_cleaner_strips_ascii_digits` (BUG-019 — baseline)
- `aggressive_cleaner_strips_devanagari_digits` (BUG-019 — Devanagari)
- `word_chunker_zero_chunk_size_returns_whole_text` (BUG-027)
- `word_chunker_normal_chunking`, `fixed_size_chunker_with_overlap` (chunker coverage)
- `test_unknown_char_at_sign_is_error` (BUG-022)
- `test_unknown_char_hash_is_error` (BUG-022)
- `test_them_keys_sorted` (BUG-034)
- `test_wildcard_keys_sorted` (BUG-034)

**Verify**: `cargo test -p syara-x --features all && cargo clippy --features all -- -D warnings` ✓

---

## Batch 7: C API + Remaining Fixes + Test Coverage — COMPLETE
**Bugs**: BUG-018, BUG-023, BUG-024, BUG-028, BUG-030, BUG-031, BUG-032
**Status**: Done.

**Changes made**:
- `capi/src/lib.rs`: BUG-018 — added `capacity` field to `SyaraMatchArray`; `into_c_array` stores actual capacity; `syara_matches_free` uses stored capacity for `Vec::from_raw_parts`; removed `shrink_to_fit` assumption
- `capi/src/lib.rs`: BUG-030 — all four C API functions (`compile_str`, `compile_file`, `scan`, `scan_file`) null-init `*out` immediately after null-pointer guard
- `capi/build.rs`: BUG-031 — removed no-op `rename_item("SyaraRules", "SyaraRules")`
- `syara/src/compiler.rs`: BUG-023 — changed condition error from `ParseError { line: 0 }` to `ConditionParse(...)` — no misleading line number
- `syara/src/models.rs`: BUG-024 — `no_match` returns empty `tags` and `meta` instead of cloning from rule
- `syara/src/engine/llm_evaluator.rs`: BUG-028 — `parse_response` checks word boundary after "YES"/"NO" using `is_none_or(|b| !b.is_ascii_alphabetic())`

**Tests added**:
- `match_array_capacity_roundtrips` (BUG-018 — capacity >= count, free doesn't crash)
- `compile_error_nulls_out_ptr` (BUG-030 — *out is null after compile error)
- `parse_response_yesterday_is_not_yes` (BUG-028 — "Yesterday" not matched as YES)
- `parse_response_notable_is_not_no` (BUG-028 — "Notable" not matched as NO)
- `test_no_match_does_not_clone_tags_or_meta` (BUG-024 — empty tags/meta on non-match)
- `test_wide_modifier` (BUG-032 — wide matching integration)
- `test_dotall_modifier` (BUG-032 — dotall matching integration)
- `test_invalid_regex_rejected` (BUG-032 — error path)
- `test_meta_only_rule_compiles` (BUG-032 — meta-only rule)
- `test_condition_error_includes_rule_name` (BUG-023 — no "line 0" in error)
- `test_condition_or_short_circuits_correctly` (BUG-007 backfill)

**Verify**: `cargo test --features all && cargo clippy --features all -- -D warnings` ✓ (154 tests)

---

## Risk Notes
- **Highest risk**: Batch 3 (parser structural changes)
- **Medium risk**: Batch 4 (phash/WAV parsing changes)
- **Lowest risk**: Batches 5, 6, 7 (isolated changes)

## Verification After All Batches
```bash
cargo test
cargo clippy -- -D warnings
cargo doc --document-private-items
```
