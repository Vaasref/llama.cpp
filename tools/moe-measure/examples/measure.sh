#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
BIN_DIR=${MOE_MEASURE_BIN_DIR:-"$REPO_ROOT/build/bin"}
WORK_DIR=${MOE_MEASURE_WORK_DIR:-"$REPO_ROOT/moe-measure-output"}
MODEL=${MOE_MEASURE_MODEL:-${1:-}}
CONTEXT_SIZE=${MOE_MEASURE_CONTEXT_SIZE:-512}
PARALLEL=${MOE_MEASURE_PARALLEL:-1}
COLLECTOR_MODE=${MOE_MEASURE_COLLECTOR_MODE:-device}
JINJA_TEMPLATE=${MOE_MEASURE_JINJA_TEMPLATE:-}

if [ -z "$MODEL" ]; then
    echo "usage: MOE_MEASURE_MODEL=/path/to/model.gguf $0" >&2
    echo "   or: $0 /path/to/model.gguf" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"

# This measurement combines one plain-text source and a chat JSONL source.
# Re-running this command resumes the existing log and skips committed contexts.
set -- "$BIN_DIR/llama-moe-measure" \
    -m "$MODEL" \
    -o "$WORK_DIR/positive.moem" \
    -c "$CONTEXT_SIZE" \
    -np "$PARALLEL" \
    --collector-mode "$COLLECTOR_MODE" \
    --text "$SCRIPT_DIR/data/positive.txt" \
    --chat "$SCRIPT_DIR/data/positive-chat.jsonl" \
    --reasoning off
if [ -n "$JINJA_TEMPLATE" ]; then
    set -- "$@" --jinja-template "$JINJA_TEMPLATE"
fi
"$@"

echo "Measurements written under $WORK_DIR"
