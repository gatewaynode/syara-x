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
pull off.  The 0.2 release will embed local LLM processing using [Burn](https://github.com/tracel-ai/burn-lm) that locally
provides a couple of LLMs for rules, this is beyond my current capability and might
be beyond Claude's.  Use at your own risk.

---

## Features

| Feature flag | Capability |
|---|---|
| _(none)_ | String/regex matching, cleaners, chunkers |
| `sbert` | Semantic similarity via HTTP embedding endpoint (Ollama) |
| `classifier` | ML text classifiers (implies `sbert`) |
| `llm` | LLM-based evaluation via Ollama `/api/chat` |
| `phash` | Perceptual hash matching for images, audio, and video |
| `all` | All of the above |

---

## Quick start

```toml
# Cargo.toml
[dependencies]
syara-x = { version = "0.1", features = ["all"] }
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

### LLM evaluation (`llm` feature)

```
rule llm_jailbreak {
    llm:
        $llm1 = {
            pattern: "Does this text attempt to override AI safety guidelines?"
            llm: ollama
            cleaner: no_op
            chunker: no_chunking
        }
    condition:
        $llm1
}
```

### Perceptual hash (`phash` feature)

```
rule known_malware_image {
    phash:
        $ph1 = {
            file_path: "/path/to/reference.png"
            threshold: 0.95
            phash: imagehash
        }
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
`ollama` (LLM), `imagehash`, `audiohash`, `videohash`

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

## License

MIT — see [LICENSE](LICENSE).
