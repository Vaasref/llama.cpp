#!/bin/sh

set -eu

SCRIPT_DIR=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
REPO_ROOT=$(CDPATH= cd -- "$SCRIPT_DIR/../../.." && pwd)
BIN_DIR=${MOE_MEASURE_BIN_DIR:-"$REPO_ROOT/build/bin"}
WORK_DIR=${MOE_MEASURE_WORK_DIR:-"$REPO_ROOT/moe-measure-output"}
MODEL=${MOE_MEASURE_MODEL:-${1:-}}
MMPROJ=${MOE_MEASURE_MMPROJ:-${2:-}}
CONTEXT_SIZE=${MOE_MEASURE_CONTEXT_SIZE:-4096}
PARALLEL=${MOE_MEASURE_PARALLEL:-1}
COLLECTOR_MODE=${MOE_MEASURE_COLLECTOR_MODE:-device}
MEDIA_PATH=${MOE_MEASURE_MEDIA_PATH:-"$REPO_ROOT/tools/mtmd"}
MMPROJ_OFFLOAD=${MOE_MEASURE_MMPROJ_OFFLOAD:-0}
SOFT_TOKEN_BUFFER_GIB=${MOE_MEASURE_SOFT_TOKEN_BUFFER_GIB:-}
DEVICES=${MOE_MEASURE_DEVICES:-}
SPLIT_MODE=${MOE_MEASURE_SPLIT_MODE:-layer}
TENSOR_SPLIT=${MOE_MEASURE_TENSOR_SPLIT:-}
JINJA_TEMPLATE=${MOE_MEASURE_JINJA_TEMPLATE:-}

if [ -z "$MODEL" ] || [ -z "$MMPROJ" ]; then
    echo "usage: MOE_MEASURE_MODEL=/path/to/model.gguf MOE_MEASURE_MMPROJ=/path/to/mmproj.gguf $0" >&2
    echo "   or: $0 /path/to/model.gguf /path/to/mmproj.gguf" >&2
    exit 1
fi

mkdir -p "$WORK_DIR"

set -- "$BIN_DIR/llama-moe-measure" \
    -m "$MODEL" \
    -mm "$MMPROJ" \
    -o "$WORK_DIR/vision-media.moem" \
    -c "$CONTEXT_SIZE" \
    -np "$PARALLEL" \
    --collector-mode "$COLLECTOR_MODE" \
    --chat "$SCRIPT_DIR/data/vision-chat.jsonl" \
    --media-path "$MEDIA_PATH" \
    --multimodal-scope media \
    --reasoning off

if [ "$MMPROJ_OFFLOAD" = "1" ]; then
    set -- "$@" --mmproj-offload
else
    set -- "$@" --no-mmproj-offload
fi
if [ -n "$DEVICES" ]; then
    set -- "$@" --device "$DEVICES" --split-mode "$SPLIT_MODE"
fi
if [ -n "$TENSOR_SPLIT" ]; then
    set -- "$@" --tensor-split "$TENSOR_SPLIT"
fi
if [ -n "$JINJA_TEMPLATE" ]; then
    set -- "$@" --jinja-template "$JINJA_TEMPLATE"
fi
if [ -n "$SOFT_TOKEN_BUFFER_GIB" ]; then
    set -- "$@" --soft-token-buffer-gib "$SOFT_TOKEN_BUFFER_GIB"
fi

"$@"

echo "Multimodal measurement written to $WORK_DIR/vision-media.moem"
