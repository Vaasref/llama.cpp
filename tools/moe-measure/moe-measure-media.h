#pragma once

#include "moe-measure-common.h"

#include <cstddef>
#include <cstdint>
#include <nlohmann/json.hpp>
#include <string>
#include <vector>

struct moe_measure_media_blob {
    std::vector<uint8_t> bytes;
    uint64_t             digest = 0;
};

bool moe_measure_media_messages_have_images(const nlohmann::ordered_json & messages,
                                     bool &                         has_images,
                                     std::string &                  error);

bool moe_measure_media_extract_images(nlohmann::ordered_json &       messages,
                               const std::string &            marker,
                               const std::string &            media_path,
                               size_t                         max_bytes,
                               std::vector<moe_measure_media_blob> & images,
                               std::string &                  error);

uint64_t moe_measure_multimodal_context_hash(const moe_measure_measurement_header &      header,
                                      const std::string &                  prompt,
                                      const std::vector<moe_measure_media_blob> & images);

uint64_t moe_measure_prefix_hash_text(uint64_t hash, int32_t token);
uint64_t moe_measure_prefix_hash_media(uint64_t hash,
                                uint64_t media_digest,
                                uint32_t chunk_index,
                                uint32_t token_index,
                                uint32_t decoder_t,
                                uint32_t decoder_x,
                                uint32_t decoder_y,
                                uint32_t decoder_z);

bool moe_measure_scope_selects(moe_measure_input_scope scope, bool media_record, bool media_position);
moe_measure_input_scope moe_measure_effective_input_scope(bool has_media,
                                            bool has_paired_media,
                                            moe_measure_input_scope requested_scope);
