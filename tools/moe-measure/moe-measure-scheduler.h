#pragma once

#include "llama.h"
#include "mtmd.h"
#include "moe-measure-common.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <vector>

enum class moe_measure_segment_kind {
    text,
    media,
};

enum class moe_measure_measure_destination {
    none,
    primary,
    prefix,
};

struct moe_measure_measure_segment {
    moe_measure_segment_kind        kind        = moe_measure_segment_kind::text;
    moe_measure_measure_destination destination = moe_measure_measure_destination::primary;
    size_t                   chunk       = 0;
    size_t                   begin       = 0;
    size_t                   count       = 0;
};

struct moe_measure_prepared_chunk {
    moe_measure_segment_kind             kind = moe_measure_segment_kind::text;
    std::vector<llama_token>      tokens;
    std::vector<float>            embeddings;
    std::vector<mtmd_decoder_pos> positions_zero;
    std::vector<mtmd_decoder_pos> positions_one;
    size_t                        n_embd     = 0;
    llama_pos                     n_pos      = 0;
    bool                          non_causal = false;
};

struct moe_measure_measure_task {
    uint64_t                          ordinal    = 0;
    bool                              multimodal = false;
    std::vector<llama_token>          tokens;
    mtmd::input_chunks                chunks;
    std::vector<moe_measure_prepared_chunk>  prepared_chunks;
    std::vector<moe_measure_measure_segment> segments;
    moe_measure_measurement_block            block;
    moe_measure_measurement_block            prefix_block;
    uint64_t                          context_hash  = 0;
    bool                              write_primary = true;
    bool                              write_prefix  = false;
    std::vector<uint8_t>              token_mask;
    std::vector<std::vector<uint8_t>> chunk_token_masks;

    explicit moe_measure_measure_task(mtmd_input_chunks * value = nullptr) : chunks(value) {}
};

struct moe_measure_measure_slot {
    llama_seq_id                       seq_id = 0;
    std::unique_ptr<moe_measure_measure_task> task;
    size_t                             segment        = 0;
    size_t                             offset         = 0;
    llama_pos                          n_past         = 0;
    size_t                             primary_output = 0;
    size_t                             prefix_output  = 0;
};

inline bool moe_measure_expand_context_size(int32_t per_slot, int32_t n_parallel, int32_t & total) {
    if (per_slot <= 0 || n_parallel <= 0 || per_slot > std::numeric_limits<int32_t>::max() / n_parallel) {
        return false;
    }
    total = per_slot * n_parallel;
    return true;
}

inline void moe_measure_init_measurement_block(moe_measure_measurement_block &        block,
                                        const moe_measure_measurement_header & header,
                                        uint64_t                        source_id,
                                        uint64_t                        context_hash,
                                        size_t                          n_tokens,
                                        bool                            allocate_observations = true) {
    block.source_id    = source_id;
    block.context_hash = context_hash;
    block.n_tokens     = n_tokens;
    block.token_hashes.resize(n_tokens);
    block.token_ids.resize(n_tokens);
    if (allocate_observations) {
        const size_t n_observations = n_tokens * header.moe_layers.size() * header.n_expert_used;
        block.expert_ids.resize(n_observations);
        block.contributions.resize(n_observations);
    }
}

inline void moe_measure_init_measurement_block(moe_measure_measure_task &             task,
                                        const moe_measure_measurement_header & header,
                                        uint64_t                        source_id,
                                        uint64_t                        context_hash,
                                        size_t                          n_tokens) {
    task.context_hash = context_hash;
    moe_measure_init_measurement_block(task.block, header, source_id, context_hash, n_tokens, false);
}

inline void moe_measure_prepare_measurement_observations(moe_measure_measurement_block &        block,
                                                  const moe_measure_measurement_header & header) {
    const size_t n_observations = block.n_tokens * header.moe_layers.size() * header.n_expert_used;
    block.expert_ids.resize(n_observations);
    block.contributions.resize(n_observations);
}

inline const mtmd_input_chunk * moe_measure_slot_chunk(const moe_measure_measure_slot & slot) {
    if (!slot.task || slot.segment >= slot.task->segments.size() || !slot.task->multimodal) {
        return nullptr;
    }
    return slot.task->chunks[slot.task->segments[slot.segment].chunk];
}

inline const moe_measure_prepared_chunk * moe_measure_slot_prepared_chunk(const moe_measure_measure_slot & slot) {
    if (!slot.task || slot.segment >= slot.task->segments.size() || !slot.task->multimodal) {
        return nullptr;
    }
    const size_t chunk = slot.task->segments[slot.segment].chunk;
    return chunk < slot.task->prepared_chunks.size() ? &slot.task->prepared_chunks[chunk] : nullptr;
}

inline uint32_t moe_measure_resolve_position(uint32_t zero, uint32_t one, llama_pos base) {
    return static_cast<uint32_t>(static_cast<int64_t>(zero) + (static_cast<int64_t>(one) - zero) * base);
}

inline bool moe_measure_soft_token_bytes(size_t n_tokens, size_t n_embd, size_t & bytes) {
    if (n_embd != 0 && n_tokens > std::numeric_limits<size_t>::max() / n_embd) {
        return false;
    }
    const size_t values = n_tokens * n_embd;
    if (values > std::numeric_limits<size_t>::max() / sizeof(float)) {
        return false;
    }
    bytes = values * sizeof(float);
    return true;
}

inline size_t moe_measure_slot_segment_tokens(const moe_measure_measure_slot & slot) {
    if (!slot.task || slot.segment >= slot.task->segments.size()) {
        return 0;
    }
    const moe_measure_measure_segment & segment = slot.task->segments[slot.segment];
    return segment.count;
}

inline bool moe_measure_slot_text_token_selected(const moe_measure_measure_slot & slot, size_t local_offset) {
    const moe_measure_measure_segment & segment = slot.task->segments[slot.segment];
    const size_t                 index   = segment.begin + local_offset;
    if (!slot.task->multimodal) {
        return slot.task->token_mask.empty() ||
               (index < slot.task->token_mask.size() && slot.task->token_mask[index] != 0);
    }
    if (segment.chunk >= slot.task->chunk_token_masks.size() || slot.task->chunk_token_masks[segment.chunk].empty()) {
        return true;
    }
    return index < slot.task->chunk_token_masks[segment.chunk].size() &&
           slot.task->chunk_token_masks[segment.chunk][index] != 0;
}

inline std::vector<size_t> moe_measure_allocate_slot_tokens(const std::vector<size_t> & remaining, size_t capacity) {
    std::vector<size_t> result(remaining.size(), 0);
    std::vector<size_t> active;
    for (size_t i = 0; i < remaining.size(); ++i) {
        if (remaining[i] > 0) {
            active.push_back(i);
        }
    }
    if (active.empty() || capacity == 0) {
        return result;
    }
    if (active.size() > capacity) {
        active.resize(capacity);
    }
    while (capacity > 0) {
        bool progressed = false;
        for (const size_t i : active) {
            if (result[i] == remaining[i]) {
                continue;
            }
            result[i]++;
            capacity--;
            progressed = true;
            if (capacity == 0) {
                break;
            }
        }
        if (!progressed) {
            break;
        }
    }
    return result;
}

inline std::vector<moe_measure_measure_slot *> moe_measure_select_text_slots(std::vector<moe_measure_measure_slot> & slots) {
    std::vector<moe_measure_measure_slot *> result;
    for (moe_measure_measure_slot & slot : slots) {
        if (slot.task && slot.segment < slot.task->segments.size() &&
            slot.task->segments[slot.segment].kind == moe_measure_segment_kind::text) {
            result.push_back(&slot);
        }
    }
    return result;
}

inline std::vector<moe_measure_measure_slot *> moe_measure_select_empty_slots(std::vector<moe_measure_measure_slot> & slots,
                                                                size_t                           available) {
    std::vector<moe_measure_measure_slot *> result;
    if (available == 0) {
        return result;
    }
    result.reserve(std::min(available, slots.size()));
    for (moe_measure_measure_slot & slot : slots) {
        if (!slot.task) {
            result.push_back(&slot);
            if (result.size() == available) {
                break;
            }
        }
    }
    return result;
}

inline bool moe_measure_consume_layer_rows(std::vector<size_t> & offsets,
                                    size_t                layer,
                                    size_t                rows,
                                    size_t                total,
                                    size_t &              begin) {
    if (layer >= offsets.size() || offsets[layer] > total || rows > total - offsets[layer]) {
        return false;
    }
    begin = offsets[layer];
    offsets[layer] += rows;
    return true;
}

inline bool moe_measure_validate_row_order(const int32_t * row_ids, size_t n_rows, size_t expected) {
    if (row_ids == nullptr || n_rows != expected) {
        return false;
    }
    std::vector<uint8_t> seen(expected, 0);
    for (size_t i = 0; i < n_rows; ++i) {
        if (row_ids[i] < 0 || static_cast<size_t>(row_ids[i]) >= expected || seen[row_ids[i]]) {
            return false;
        }
        seen[row_ids[i]] = 1;
    }
    return true;
}

inline bool moe_measure_append_row_order(
        const int32_t * row_ids, size_t n_rows, size_t expected, std::vector<uint8_t> & seen) {
    if (row_ids == nullptr || seen.size() != expected || n_rows > expected) {
        return false;
    }
    for (size_t i = 0; i < n_rows; ++i) {
        if (row_ids[i] < 0 || static_cast<size_t>(row_ids[i]) >= expected || seen[row_ids[i]]) {
            return false;
        }
        seen[row_ids[i]] = 1;
    }
    return true;
}

inline bool moe_measure_decode_captured_expert(float encoded, uint32_t n_experts, int32_t & expert) {
    const double value = encoded;
    if (!std::isfinite(encoded) || std::trunc(encoded) != encoded || value < 0.0 || value >= n_experts) {
        return false;
    }
    expert = static_cast<int32_t>(encoded);
    return true;
}
