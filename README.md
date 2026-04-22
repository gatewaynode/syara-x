# syara-x

**Semantic YARA in Rust** — extends YARA-compatible rules with semantic similarity,
ML classifier, LLM-based, and perceptual hash matching. Catches malicious
content (prompt injection, phishing, jailbreaks) by meaning and intent, not
just exact text patterns.

Ported from [SYARA](https://github.com/nabeelxy/syara) originally written by
Nabeel Yoosuf

> This library was ported from Python to Rust by
> [Claude](https://claude.ai/claude-code) (Anthropic's AI coding assistant),
> working through six implementation phases under human direction, tested in a
> larger implementation that includes working side by side with [YARA-X](https://github.com/VirusTotal/yara-x).
> See [CONTRIBUTING.md](CONTRIBUTING.md) for how the project is maintained.

**EXPERIMENTAL**: Do not use this for anything important yet.  I'm lazily commmiting
directly to `main` some very speculative features that I'm not sure if Claude can
pull off.  The local-LLM backend (`burn-llm` / `burn-llm-gpu`) is currently walled
off pending a migration to `candle-rs` — see [ROADMAP.md](ROADMAP.md).  Use at your
own risk.

---

## Features

| Feature flag | Capability |
|---|---|
| _(none)_ | String/regex matching, cleaners, chunkers |
| `sbert` | Semantic similarity via HTTP embedding endpoint (OpenAI-compatible; Ollama variant preserved) |
| `sbert-onnx` | Local ONNX MiniLM-L6-v2 backend (requires system `libonnxruntime` ≥1.17 — see [System dependencies](#system-dependencies)) |
| `classifier` | ML text classifiers via OpenAI-compatible HTTP embeddings (implies `sbert`) |
| `classifier-onnx` | Local ONNX classifier backend (recommended; implies `classifier` + `sbert-onnx`) |
| `llm` | LLM-based evaluation via OpenAI-compatible `/v1/chat/completions` (LM Studio / vLLM / llama-server / openai.com / Ollama's shim); native Ollama `/api/chat` preserved as legacy |
| `phash` | Perceptual hash matching for images, audio, and video |
| `all` | All of the above |

---

## Quick start

```toml
# Cargo.toml
[dependencies]
syara-x = { version = "0.3", features = ["all"] }
```

```rust
use syara_x;

let rules = syara_x::compile_str(r#"
    rule prompt_injection {
        strings:
            $pi1 = "ignore previous instructions" nocase
            $pi2 = "disregard your system prompt" nocase
        condition:
            any of them
    }
"#)?;

for m in rules.scan(user_input) {
    if m.matched {
        println!("Rule '{}' matched", m.rule_name);
    }
}
```

---

## Rule syntax

syara-x uses a YARA-inspired DSL with extensions for semantic and ML matching.

### String patterns

```
rule example {
    strings:
        $s1 = "literal match" nocase
        $s2 = /regex\s+pattern/
        $s3 = "wide char" wide
    condition:
        $s1 or $s2
}
```

Supported modifiers: `nocase`, `wide`, `ascii`, `dotall`, `fullword`.

Regex patterns support Rust's inline flag syntax — `(?m)^foo` (multiline),
`(?s).*` (dot-matches-newline), `(?i)foo` (case-insensitive). Inline flags
compose with modifiers.

### Condition expressions

Conditions are boolean expressions over pattern identifiers (`$name`) and
pattern match counts (`#name`). YARA-style:

```
rule multi_turn_transcript {
    strings:
        $user      = "user:"
        $assistant = "assistant:"
    condition:
        #user >= 2 and #assistant >= 1
}
```

Supported: `$id`, `#id` (count), integer literals, `== != < <= > >=`,
`+` / `-` arithmetic, unary `-`, `and` / `or` / `not`, `any of` / `all of`
pattern sets (`(them | $a,$b | $prefix*)`). See
[tasks/YARA-X-PARITY-GAPS.md](tasks/YARA-X-PARITY-GAPS.md) for what's not
yet supported vs YARA-X.

### Semantic similarity (`sbert` feature)

```
rule semantic_phishing {
    similarity:
        $sim1 = {
            pattern: "your account has been compromised click here"
            threshold: 0.82
            cleaner: default_cleaning
            chunker: sentence_chunking
            matcher: sbert
        }
    condition:
        $sim1
}
```

### Classifier (`classifier` / `classifier-onnx` features)

```
rule jailbreak_classifier {
    classifier:
        $c1 = {
            pattern: "request to override AI safety guidelines"
            threshold: 0.65
            cleaner: default_cleaning
            chunker: paragraph_chunking
            classifier: tuned-sbert
        }
    condition:
        $c1
}
```

The default `tuned-sbert` classifier is registered against an OpenAI-compatible
`/v1/embeddings` endpoint (`http://localhost:1234`). For deterministic, offline
scoring use the local ONNX backend instead:

```rust
use syara_x::engine::classifier::OnnxEmbeddingClassifier;
let cls = OnnxEmbeddingClassifier::from_dir("../models/all-MiniLM-L6-v2")?;
rules.register_classifier("tuned-sbert", Box::new(cls));
```

### LLM evaluation (`llm` feature)

```
rule llm_jailbreak {
    llm:
        $llm1 = {
            pattern: "Does this text attempt to override AI safety guidelines?"
            llm: openai-api-compatible
            cleaner: no_op
            chunker: no_chunking
        }
    condition:
        $llm1
}
```

`openai-api-compatible` is the default and talks to any OpenAI-compatible
`/v1/chat/completions` endpoint (LM Studio, vLLM, llama-server, Open WebUI,
openai.com, and Ollama's OpenAI-compat shim). The legacy `ollama` name is kept
for users on Ollama's native `/api/chat` endpoint — specify `llm: ollama`.

#### Default endpoint and environment variables

The default registration points at `http://localhost:1234/v1/chat/completions`
with model `local-model` — the LM Studio convention. Override via environment:

| Variable | Precedence | Purpose |
|---|---|---|
| `SYARA_LLM_ENDPOINT` | highest | Full chat-completions URL (scoped to SYARA-X) |
| `SYARA_LLM_MODEL` | highest | Model identifier |
| `SYARA_LLM_API_KEY` | highest | Bearer token |
| `OPENAI_BASE_URL` | fallback | Root URL; `/chat/completions` is appended if missing |
| `OPENAI_MODEL` | fallback | Model identifier |
| `OPENAI_API_KEY` | fallback | Bearer token |

**Opting out of env-var lookup.** If you prefer SYARA-X never read these vars,
set `SYARA_LLM_NO_ENV=1` — the default registration falls back to the
hardcoded localhost endpoint with no API key. Alternatively, register an
explicit evaluator before scanning:

```rust
# #[cfg(feature = "llm")] {
use syara_x::engine::llm_evaluator::OpenAiChatEvaluatorBuilder;
let mut rules = syara_x::compile_str(source)?;
rules.register_llm_evaluator(
    "openai-api-compatible",
    Box::new(
        OpenAiChatEvaluatorBuilder::new()
            .endpoint("http://localhost:1234/v1/chat/completions")
            .model("local-model")
            .api_key(std::fs::read_to_string("/run/secrets/api_key")?.trim())
            .build(),
    ),
);
# Ok::<(), syara_x::SyaraError>(()) }
```

**Scoped token exposure.** Prefer `SYARA_LLM_API_KEY` over `OPENAI_API_KEY`
to keep SYARA-X's credential separate from anything else on the system that
reads `OPENAI_API_KEY`. A process-scoped invocation also works:
`SYARA_LLM_API_KEY="$(cat /path/to/key)" your-binary`.

#### Builder knobs

[`OpenAiChatEvaluatorBuilder`] exposes `endpoint`, `model`, `api_key`,
`temperature` (default `0.0` — deterministic), `max_tokens` (default 8192 —
sized for reasoning models like Qwen3, DeepSeek-R1, GPT-OSS that spend
thousands of tokens on internal `<think>` / `reasoning_content` before the
final `YES`/`NO`; non-reasoning models still stop early via
`finish_reason=stop`), `system_prompt`, `header` (arbitrary HTTP header),
`connect_timeout` (default 20s), and `read_timeout` (default 60s).

If a reasoning model is truncated mid-think (`finish_reason=length` with
empty `content`), the evaluator returns a `SyaraError::LlmError` pointing
at `.max_tokens(…)` rather than silently parsing the empty response as a
non-match.

Responses are cached by `(pattern, chunk)` when `temperature == 0.0`; the
cache is bounded (1024 entries) and cleared after each `scan()` call to
mirror the lifecycle of the text cache.

### Perceptual hash (`phash` feature)

```
rule known_malware_image {
    phash:
        $ph1 = "/path/to/reference.png" threshold=0.95 hasher="imagehash"
    condition:
        $ph1
}
```

---

## Built-in components

**Cleaners:** `default_cleaning`, `aggressive_cleaning`, `no_op`

**Chunkers:** `no_chunking`, `sentence_chunking`, `paragraph_chunking`,
`word_chunking`, `fixed_size_chunking`

**Matchers:** `sbert` (HTTP embedding), `tuned-sbert` (classifier),
`openai-api-compatible` (LLM, default), `ollama` (LLM, legacy),
`imagehash`, `audiohash`, `videohash`

Custom components can be registered on `CompiledRules` via
`register_cleaner`, `register_chunker`, `register_semantic_matcher`, etc.

---

## C API

A C FFI is available via the `capi` crate. After building, `syara_x.h` is
generated automatically by [cbindgen](https://github.com/mozilla/cbindgen).

```c
#include "syara_x.h"

SyaraRules *rules = NULL;
syara_compile_str("rule r { strings: $s = \"evil\" condition: $s }", &rules);

SyaraMatchArray *matches = NULL;
syara_scan(rules, input_text, &matches);

for (size_t i = 0; i < matches->count; i++) {
    if (matches->matches[i].matched) {
        printf("matched: %s\n", matches->matches[i].rule_name);
    }
}

syara_matches_free(matches);
syara_rules_free(rules);
```

---

## Architecture

```
.syara file
    └─> SyaraParser     parse DSL
    └─> Compiler        validate identifiers, conditions
    └─> CompiledRules   execution engine
            ├─ StringMatcher     (cheapest)
            ├─ SemanticMatcher   (sbert)
            ├─ PHashMatcher      (phash)
            ├─ TextClassifier    (classifier)
            └─ LLMEvaluator      (most expensive, short-circuited)
```

Execution is cost-ordered. LLM calls are skipped when the condition cannot
be satisfied even if the LLM matches (see `is_identifier_needed` in
`condition.rs`).

---

## Development

```bash
cargo build                          # build all crates
cargo test                           # run all tests
cargo test -p syara-x --features all # library tests with all features
cargo clippy -- -D warnings          # lint (must be clean)
```

External services (Ollama) are only contacted when the corresponding feature
is enabled and a rule actually exercises that matcher. String-only rules need
no external services.

---

## System dependencies

Most features are pure-Rust and need nothing beyond `cargo`. The `sbert-onnx`
and `classifier-onnx` features are the exceptions — both link against the ONNX
Runtime shared library at runtime (via `ort`'s `load-dynamic` mode) and will
not run without it installed on the host.

### `sbert-onnx` / `classifier-onnx` (ONNX Runtime ≥ 1.17)

**macOS (Homebrew):**

```bash
brew install onnxruntime
# Homebrew installs to /opt/homebrew/lib, which dlopen does NOT search by default —
# point ort at the dylib explicitly:
export ORT_DYLIB_PATH="$(brew --prefix onnxruntime)/lib/libonnxruntime.dylib"
```

**Linux (Debian/Ubuntu):** download the matching release from
[microsoft/onnxruntime releases](https://github.com/microsoft/onnxruntime/releases)
and place `libonnxruntime.so` on your loader path, or set `ORT_DYLIB_PATH` to
point at the file.

**Any platform (escape hatch):** point `ort` at a specific dylib by exporting
`ORT_DYLIB_PATH=/absolute/path/to/libonnxruntime.{dylib,so,dll}` before
`cargo test` / `cargo run`.

**Convenience wrapper:** for repeated use, run
`./scripts/install_onnxruntime_xdg.sh` once to install
`~/.local/bin/with-onnxruntime` (XDG user-bin). Then prefix any command:

```bash
with-onnxruntime cargo test --features classifier-onnx -- --ignored
```

The same MiniLM weights are reused by `integration_real_onnx_embed` and
`integration_real_onnx_classifier`. To fetch them:

```bash
./scripts/fetch_minilm.sh       # downloads to <repo>/models/all-MiniLM-L6-v2/
```

---

## License

MIT — see [LICENSE](LICENSE).
