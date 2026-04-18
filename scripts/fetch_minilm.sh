#!/usr/bin/env bash
#
# Fetch sentence-transformers/all-MiniLM-L6-v2 for the `sbert-onnx` feature.
#
# Places `model.onnx` + `tokenizer.json` at the root of the target directory
# (matching the layout expected by `OnnxEmbeddingMatcher::from_dir`).
#
# Default target: <repo-root>/models/all-MiniLM-L6-v2/   (mirrors Phase 5 convention
# — cargo test sets CWD to the `syara/` package dir, so `../models/` there
#   resolves to <repo-root>/models/.)
# Override with:  MINILM_DIR=/path/to/dir ./scripts/fetch_minilm.sh

set -euo pipefail

REPO="sentence-transformers/all-MiniLM-L6-v2"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
TARGET="${MINILM_DIR:-$REPO_ROOT/models/all-MiniLM-L6-v2}"

mkdir -p "$TARGET"

if command -v huggingface-cli >/dev/null 2>&1; then
    echo "==> Using huggingface-cli to download into $TARGET"
    # Pull only the two files we need; `onnx/model.onnx` gets rewritten below.
    huggingface-cli download "$REPO" \
        onnx/model.onnx \
        tokenizer.json \
        --local-dir "$TARGET"
    if [ -f "$TARGET/onnx/model.onnx" ] && [ ! -f "$TARGET/model.onnx" ]; then
        mv "$TARGET/onnx/model.onnx" "$TARGET/model.onnx"
        rmdir "$TARGET/onnx" 2>/dev/null || true
    fi
else
    echo "==> huggingface-cli not found; falling back to curl"
    BASE="https://huggingface.co/${REPO}/resolve/main"
    curl -fL --retry 3 -o "$TARGET/model.onnx"     "$BASE/onnx/model.onnx"
    curl -fL --retry 3 -o "$TARGET/tokenizer.json" "$BASE/tokenizer.json"
fi

echo "==> Done. Files in $TARGET:"
ls -lh "$TARGET/model.onnx" "$TARGET/tokenizer.json"
