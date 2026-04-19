# Task Refinement — HTTP LLM Provider Configurability

**Task:** Update the HTTP interface to LLM providers for more configurability and support for local inference engines like LM Studio (preferred engine in this environment).

**Current state of `syara/src/engine/llm_evaluator.rs`:**
- Single `OllamaEvaluator` struct targeting Ollama's `/api/chat` wire format
- Hardcoded defaults: `http://localhost:11434/api/chat`, model `llama3.2`, timeouts 10s connect / 30s read
- No API key, temperature, max_tokens, or system-prompt configurability
- Meanwhile `sbert` and `classifier` already default to the OpenAI-compatible `/v1/embeddings` wire format on LM Studio port 1234 (`config.rs:86-101`)

Answer inline under each question. Leave blank to accept the "lean" / default.

---

## 1. Scope — add a new backend or replace Ollama?

OpenAI-compatible `/v1/chat/completions` covers LM Studio, vLLM, llama-server, Open WebUI, openai.com, and Ollama (via its OpenAI-compat endpoint).

- **(a)** Add `OpenAiChatEvaluator` alongside `OllamaEvaluator`; change registry default to the OpenAI-compat one (pointed at LM Studio port 1234). Ollama stays for users on its native endpoint. *(mirrors what we did for sbert — non-breaking)*
- **(b)** Replace `OllamaEvaluator` with `OpenAiChatEvaluator` entirely (breaking change; Ollama users switch to its `/v1/chat/completions` endpoint).

**My lean:** (a)

**Your answer:**  Agreed.  Add the `OpenAIChatEvaluator` along side the `OllamaEvaluator`.  More options will make this library more useful as long as we can manage the complexity.

---

## 2. Configurability — which knobs do you want exposed?

Proposed (via builder, e.g. `OpenAiChatEvaluatorBuilder`):

- [ ] `endpoint` (required)
- [ ] `model` (required)
- [ ] `api_key` — Bearer token, optional for local
- [ ] `temperature` — default `0.0` (deterministic matching)
- [ ] `max_tokens` — default ~128
- [ ] `connect_timeout` — default 10s
- [ ] `read_timeout` — default 30s
- [ ] `system_prompt` override
- [ ] Custom extra headers (for proxies / auth schemes)

**Anything to add, drop, or change defaults on?**

**Your answer:**  I like the proposed knobs, we can definitely start there.  The connect and read default timeouts seem a little low, I'd recommend doubling them.

---

## 3. Env-var auto-detection?

Should the default registered evaluator pick up `OPENAI_API_KEY` / `OPENAI_BASE_URL` / `OPENAI_MODEL` from env at construction time?

Caveat: sbert/classifier currently don't do env lookup — introducing it in one place is a small consistency break (or a good reason to add it to those too).

**My lean:** Yes, but only for the *default* registration (explicit `new()` calls stay explicit). Document that it also applies to sbert/classifier only if/when the user asks.

**Your answer:**  Env-var auto-detection is a sensitive subject, developers can have far more scope than they are comfortable with sharing in the env-vars.  Let's make it configurable and default to `read_env-vars = true`, but update the README.md to be explicit about the concerns with clear directions on how to disable env-var reading and how to discretely pass just the security tokens that SYARA-X needs.  And I agree this should be a read once and all features should be able to access it as needed.

---

## 4. Registry default name

Currently registered as `"ollama"`. For the new default, I'd register the OpenAI-compat evaluator as `"llm"` (generic) and keep `"ollama"` for the legacy native endpoint.

- Rules in `.syara` files that reference `llm="ollama"` still work
- New rules use `llm="llm"` or omit and get the default

**My lean:** register new default as `"llm"`; keep `"ollama"` for legacy.

**Alternative names if you prefer:** `"openai"`, `"chat"`, `"local"`

**Your answer:** Yes, our OpenAI API option should be the new default and Ollama the legacy.  I think the default naming should be a bit more descriptive, so `openai-api-compatable` would be more fitting.

---

## 5. Response caching?

sbert/classifier share a response cache (`HttpEmbedder`). Should LLM YES/NO responses also be cached (keyed on pattern + chunk hash)?

- Makes sense with `temperature=0.0` (deterministic)
- Skip if we expect non-deterministic temperature in practice
- Caveat: the per-scan `TextCache` is already cleared each scan; cross-scan caching would be a new behavior

**My lean:** add a bounded response cache keyed on `(pattern, chunk)` when `temperature==0.0`, cleared per scan to match existing cache lifecycle. Skip if that's overreach.

**Your answer:** I think caching is pretty risky for a risk scanner, but performance is very important too.  Your "lean" makes a nice compromise that I think is worth pursuing.

---

## 6. Real-model test

Add `integration_real_openai_chat` `#[ignore]` test hitting LM Studio on 1234, matching the existing `integration_real_openai_embed` pattern?

**My lean:** yes. Also update `CLAUDE.md`'s real-model-tests block with the new command.

**Your answer:**  Yes, let's do real world testing hitting the LMstudio endpoint.  Caveat, the README.md should use generic LMstudio examples, our LMStudio is serving on `http://169.254.176.134:1234` for some reason and has the models `qwen/qwen3.6-35b-a3b` and `minimax/minimax-m2.7` loaded and available.

---

## 7. Anything else?

Open-ended — anything else you want in scope or explicitly out of scope? E.g. streaming, retries, non-chat completions endpoint, token-level telemetry, JSON-mode / structured output...

**Your answer:** I mean, yes there will be more refinement needed.  But let's not get ahead of ourselves.  Let's get this working first and then we can contemplate additional features and quality of life improvements.
