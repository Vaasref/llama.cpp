# MoE expert measurement

This private-fork tool measures expert activity in standard stacked-expert
GGUF MoE models. Mistral Small 4 is the primary supported layout. Every model is checked
from its GGUF metadata before work starts; unfamiliar expert layouts are rejected.

## Build

```sh
cmake -B build
cmake --build build --target llama-moe-measure
```

Runnable examples are under [`examples/`](examples/):

```sh
tools/moe-measure/examples/measure.sh /path/to/model.gguf
tools/moe-measure/examples/measure-multimodal.sh /path/to/model.gguf /path/to/mmproj.gguf
```

They default to `moe-measure-output/` at the repository root. The scripts also accept
`MOE_MEASURE_MODEL`, `MOE_MEASURE_BIN_DIR`, `MOE_MEASURE_WORK_DIR`, `MOE_MEASURE_CONTEXT_SIZE`, `MOE_MEASURE_JINJA_TEMPLATE`,
`MOE_MEASURE_PARALLEL`, and `MOE_MEASURE_COLLECTOR_MODE`.
The multimodal script keeps the projector on CPU by default. Set `MOE_MEASURE_MMPROJ_OFFLOAD=1`
to offload it, or set `MOE_MEASURE_DEVICES`, `MOE_MEASURE_SPLIT_MODE`, and `MOE_MEASURE_TENSOR_SPLIT` to control
language-model placement.

For the two-device Mistral Small 4 setup used by `start_router.sh`:

```sh
HIP_VISIBLE_DEVICES=0,2 \
GGML_VK_VISIBLE_DEVICES=1,2 \
GGML_VK_SUBALLOCATION_BLOCK_SIZE=6489335280 \
MOE_MEASURE_DEVICES=Vulkan0,Vulkan1 \
MOE_MEASURE_SPLIT_MODE=layer \
MOE_MEASURE_TENSOR_SPLIT=5,3 \
tools/moe-measure/examples/measure-multimodal.sh \
    models_/Mistral-Small-4-119B-2603-UD-IQ2_XXS.gguf \
    models_/Mistral_Small_4_mmproj-F32.gguf
```

## Measure

New measurement logs conventionally use `.moem`. The extension is not validated, and
version-5 `.reapm` logs produced by the earlier private-fork tool remain readable and
resumable because the on-disk format is unchanged.

```sh
build/bin/llama-moe-measure \
    -m model.gguf -o general.moem -c 512 \
    --text prose-1.txt --text prose-2.txt \
    --chat conversations.jsonl
```

`--input-jsonl` measures a required `input` while retaining an optional
`context_prefix` for conditioning. Each part can independently be a raw string or an
OpenAI-compatible message array:

```json
{"context_prefix":"Question: Capital of France?\nAnswer: ","input":"Paris"}
{"context_prefix":[{"role":"user","content":"Capital of France?"}],"input":"Paris"}
{"context_prefix":[{"role":"user","content":"Question"}],"input":[{"role":"assistant","content":"Answer"}]}
```

The primary `-o` file receives only input-owned positions. Add `--prefix-output` to
write retained prefix-owned positions from the same decode to a second append-only log:

```sh
build/bin/llama-moe-measure \
    -m model.gguf -o target.moem --prefix-output prefix.moem \
    --input-jsonl examples.jsonl
```

No separator is inserted between string parts. A message prefix followed by a string is
rendered with a generation prompt before the bytes are appended. Two message arrays are
combined and rendered once. A raw prefix followed by messages prepends the raw bytes to
the rendered messages. Ownership is determined after tokenizing the combined prompt: the
longest token prefix shared with the separately rendered prefix belongs to the prefix, and
the first token changed by the boundary belongs to the input. Template wrappers follow
that rule.

If a paired record is too large, only the left side of `context_prefix` is removed. The tool keeps
the longest suffix that fits both decoder-token and multimodal-position limits,
preserves BOS when possible, and never splits a projected image chunk. The complete input
must fit. Images can occur in either message-array part and are written to the owning
output. Paired input records measure both text and projected image positions; vocabulary
filters affect only text.

Use repeatable token-ID files to filter language-token observations:

```sh
build/bin/llama-moe-measure \
    -m model.gguf -o target.moem --input-jsonl examples.jsonl \
    --measured-vocab allow.txt --excluded-vocab deny.txt \
    --exclude-special-tokens
```

Files contain one decimal token ID per line. Blank lines and `#` comments are ignored.
Allow files form a union; without one, all vocabulary IDs start enabled. Deny files form a
union and override the allowlist. `--exclude-special-tokens` denies control, EOG,
user-defined, unknown, and unused tokens while leaving normal and byte tokens eligible.
The mask applies to `--text`, `--chat`, and paired inputs, but rejected tokens are still
decoded causally. Projected image rows are never filtered. Special-token parsing is
enabled by default, so externally rendered model-specific BOS, EOS, and chat-control
spellings tokenize as their special IDs when the vocabulary defines them.
`--no-parse-special` is available only for compatibility or tokenizer diagnosis. A
spelling such as `</s>` is not universal: if the model does not define it as a special
token, it remains ordinary text. Template artifacts represented by normal pieces need an
explicit denylist.

Text files are joined with newline separators and tokenized once. Chat files are JSONL;
each non-empty line must contain an OpenAI-compatible `messages` array and may contain
`tools`, `chat_template_kwargs`, `enable_thinking`, and `add_generation_prompt`. The last
option defaults to false.

Image-bearing records accept OpenAI-compatible `image_url` parts. Use either a
`data:image/...;base64,...` URI or a `file://relative/path` under the canonical
`--media-path` directory, and provide `-mm MMPROJ.gguf`. HTTP URLs, raw base64, path
escapes, audio, and video are rejected. Each image record is one independent context and
is rejected if its decoded positions exceed the active context.

`--multimodal-scope media|text|all` controls which positions are recorded for an
image-bearing record. The default `media` records only the projected image soft tokens;
`text` records template and boundary text while retaining image conditioning; `all`
records both. Text-only records always measure their text. Projector preprocessing is
configured with `--image-min-tokens`, `--image-max-tokens`, and
`--mtmd-batch-max-tokens`; `--media-max-bytes` defaults to 64 MiB per encoded image.
`--soft-token-buffer-gib` limits buffered F32 projector embeddings and defaults to 10% of
physical RAM, or 1 GiB when RAM detection is unavailable. One context larger than the
limit is processed alone.

The tool inspects JSONL record structure before reading projector metadata. If no record
contains media, multimodal options are inactive: the projector is not inspected or
loaded, no soft-token buffer is allocated, and the measurement uses canonical text
metadata. A text-only paired dataset does not force the measurement scope to `all`.
Vocabulary filters still apply because text-only records measure language tokens.

Input preprocessing happens before full model loading. Text-only runs tokenize all
pending contexts and load the full model once. Multimodal runs project a bounded group of
contexts, unload the MMProj, load the full model to measure that group, and repeat. The
MMProj and full model weights are not resident at the same time. The vocabulary-only
model used for tokenization and templates remains loaded throughout.

The embedded model template is used by default. To use a specific Jinja2 file, pass
`--jinja-template /path/to/template.jinja`. This reads the template, enables the Jinja
engine, and includes the rendered template configuration in measurement resume identity.
It cannot be combined with `--no-jinja` or another template override. The older
`--chat-template`, `--chat-template-file`, `--chat-template-kwargs`, `--jinja`/`--no-jinja`,
and `--reasoning on|off|auto` options remain available.

```sh
build/bin/llama-moe-measure \
    -m model.gguf -o chat.moem -c 4096 \
    --chat conversations.jsonl \
    --jinja-template templates/model-chat-template.jinja \
    --chat-template-kwargs '{"enable_thinking": false}'
```

Measurement progress is reported at info level. It identifies model and projector loading,
the selected template, dataset and record preparation, token/context counts, resume skips,
multimodal preprocessing, periodic commits, and a final timing and transfer summary. Per-batch
decode and slot-assignment messages are available at debug level. Standard llama.cpp log options
can redirect or suppress this output.

Use `-np N` to measure up to `N` independent contexts in parallel through one batched model
decode. Unlike the server, the tool treats `-c` as the context size of each measurement slot and
allocates approximately `-c x -np` total KV capacity. `-b` and `-ub` remain global limits for
each physical decode, so both may need to be increased to keep several slots busy:

```sh
build/bin/llama-moe-measure \
    -m model.gguf -o chat.moem --chat conversations.jsonl \
    -c 4096 -np 4 -b 2048 -ub 2048
```

Parallel slots increase KV, prepared-media, and temporary measurement memory. They do not
change measurement identity, and an existing single-slot `.moem` can be resumed with a
different slot count when the per-slot context and other measurement settings are unchanged.

`--collector-mode device` is the default. It reduces expert-output norms on the model backend,
does not evaluate the unused vocabulary head, and transfers only norms, router weights, and
expert IDs to the host. `--collector-mode cpu` retains the full-output collector as a diagnostic
fallback. Both modes implement the same metric and can resume the same measurement log; small
floating-point differences from reduction order are expected.

The default metric for a selected expert and token is:

```text
(router_weight / sum(selected_router_weights)) * L2(unweighted_expert_output)
```

`--router-weights model` preserves the model's final selected weights instead. Files made
with different metric modes cannot be resumed together.

Measurements are append-only `.moem` logs. Each context is committed independently with
a checksum and stable FNV-1a context and token-prefix hashes. Starting the same command
again scans the log, removes an incomplete final block if present, and does not evaluate
contexts whose hashes are already committed. Complete blocks remain sparse, ordered, and
unaggregated. If all contexts are already committed, preprocessing completes without
loading the full model. Committed non-paired image records are also skipped before loading
the MMProj; paired multimodal records still require projector preprocessing to determine
their ownership boundary and hashes.

Measurement format version 5 includes an output role, final vocabulary-mask hash, and the
token ID for every measured decoder token. Text token IDs use the model vocabulary;
projected media positions use `-1` because they do not correspond to vocabulary tokens.
Token IDs are stored in the same order as the per-token prefix hashes. For token `t`, its
expert observations start at:

```text
(t * moe_layer_count + layer_index) * expert_used_count
```

After the block's 32-byte fixed fields, the version 5 payload arrays are:

```text
uint64 token_hashes[n_tokens]
int32  token_ids[n_tokens]
uint32 expert_ids[n_tokens * moe_layer_count * expert_used_count]
float  contributions[n_tokens * moe_layer_count * expert_used_count]
```

Storing token IDs can reveal substantially more about the measured input than prefix
hashes alone. Treat measurement files as potentially reconstructable input data.

The output role and vocabulary-mask hash prevent a prefix log from being resumed as a
primary log and prevent filtered and unfiltered runs from sharing resume state. Older
measurement formats are rejected and must be regenerated.
For non-uniform MoE models, the header expert count is the maximum across MoE layers.
Blocks remain rectangular and routed IDs are validated against the actual count of their
layer, so the per-token observation layout does not vary by layer.

Measurements use a canonical model signature based on architecture, MoE metadata,
tensor names, and logical tensor shapes. Storage types and shard placement are excluded, so
logs from equivalent quantizations or shard layouts share the same model identity.
Measurement headers bind resume identity to the projector structure, media
configuration, scope, and media pipeline version. Image context hashes include the
templated prompt and ordered encoded image bytes, so a committed record is skipped before
bitmap preprocessing, projector encoding, and language-model decoding.
