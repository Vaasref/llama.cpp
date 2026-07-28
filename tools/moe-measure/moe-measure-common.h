#pragma once

#include "ggml.h"

#include <cstdint>
#include <functional>
#include <string>
#include <unordered_set>
#include <vector>

struct moe_measure_id {
    uint64_t hi = 0;
    uint64_t lo = 0;

    bool operator==(const moe_measure_id & other) const { return hi == other.hi && lo == other.lo; }
};

struct moe_measure_model_info {
    std::string              architecture;
    uint64_t                 signature           = 0;
    uint32_t                 n_layer             = 0;
    uint32_t                 n_embd              = 0;
    uint32_t                 n_expert            = 0;
    uint32_t                 n_expert_used       = 0;
    uint32_t                 n_expert_group      = 0;
    uint32_t                 n_expert_group_used = 0;
    uint32_t                 n_split             = 1;
    std::vector<uint32_t>    n_expert_per_layer;
    std::vector<int32_t>     moe_layers;
    std::vector<std::string> split_paths;
};

enum moe_measure_input_scope : uint32_t {
    MOE_MEASURE_INPUT_SCOPE_UNKNOWN = 0,
    MOE_MEASURE_INPUT_SCOPE_TEXT    = 1,
    MOE_MEASURE_INPUT_SCOPE_MEDIA   = 2,
    MOE_MEASURE_INPUT_SCOPE_ALL     = 3,
};

enum moe_measure_router_metric : int8_t {
    MOE_MEASURE_ROUTER_METRIC_UNKNOWN      = -1,
    MOE_MEASURE_ROUTER_METRIC_MODEL        = 0,
    MOE_MEASURE_ROUTER_METRIC_RENORMALIZED = 1,
};

enum moe_measure_output_role : uint32_t {
    MOE_MEASURE_OUTPUT_ROLE_UNKNOWN = 0,
    MOE_MEASURE_OUTPUT_ROLE_PRIMARY = 1,
    MOE_MEASURE_OUTPUT_ROLE_PREFIX  = 2,
};

struct moe_measure_measurement_header {
    uint32_t             format_version = 5;
    std::string          architecture;
    uint64_t             model_signature = 0;
    moe_measure_id              measurement_id;
    uint64_t             template_hash             = 0;
    uint64_t             tokenization_hash         = 0;
    uint32_t             n_layer                   = 0;
    uint32_t             n_expert                  = 0;
    uint32_t             n_expert_used             = 0;
    uint32_t             n_ctx                     = 0;
    bool                 router_weights_normalized = true;
    uint64_t             projector_signature       = 0;
    uint64_t             media_config_hash         = 0;
    uint32_t             media_pipeline_version    = 0;
    moe_measure_input_scope     input_scope               = MOE_MEASURE_INPUT_SCOPE_TEXT;
    moe_measure_output_role     output_role               = MOE_MEASURE_OUTPUT_ROLE_PRIMARY;
    uint64_t             vocab_mask_hash           = 0;
    std::vector<int32_t> moe_layers;
};

struct moe_measure_measurement_block {
    uint64_t              source_id    = 0;
    uint64_t              context_hash = 0;
    uint32_t              n_tokens     = 0;
    std::vector<uint64_t> token_hashes;
    std::vector<int32_t>  token_ids;
    std::vector<uint32_t> expert_ids;
    std::vector<float>    contributions;
};

struct moe_measure_measurement_summary {
    moe_measure_measurement_header      header;
    uint64_t                     content_digest   = 0;
    uint64_t                     valid_size       = 0;
    uint64_t                     n_blocks         = 0;
    bool                         has_partial_tail = false;
    bool                         has_text_tokens  = false;
    bool                         has_media_tokens = false;
    std::unordered_set<uint64_t> context_hashes;
};

struct moe_measure_measurement_header_mismatch {
    std::string field;
    std::string requested;
    std::string existing;
};


uint64_t    moe_measure_hash_bytes(uint64_t hash, const void * data, size_t size);
uint64_t    moe_measure_hash_string(uint64_t hash, const std::string & value);
uint64_t    moe_measure_hash_tokens(const std::vector<int32_t> & tokens);
moe_measure_id     moe_measure_make_id(const std::string & salt);
std::string moe_measure_id_string(const moe_measure_id & id);
bool        moe_measure_calculate_contribution(double  router_weight,
                                        double  selected_weight_sum,
                                        double  expert_output_norm_squared,
                                        bool    normalize_router_weight,
                                        float & contribution);

bool moe_measure_model_info_load(const std::string & path, moe_measure_model_info & info, std::string & error);
bool moe_measure_gguf_structural_signature(const std::string & path, uint64_t & signature, std::string & error);

bool moe_measure_measurement_create(const std::string & path, const moe_measure_measurement_header & header, std::string & error);
bool moe_measure_measurement_read(const std::string &                                         path,
                           moe_measure_measurement_summary &                                  summary,
                           const std::function<bool(const moe_measure_measurement_block &)> & callback,
                           std::string &                                               error);
bool moe_measure_measurement_truncate_tail(const std::string &              path,
                                    const moe_measure_measurement_summary & summary,
                                    std::string &                    error);
bool moe_measure_measurement_append(const std::string &             path,
                             const moe_measure_measurement_header & header,
                             const moe_measure_measurement_block &  block,
                             std::string &                   error);
std::vector<moe_measure_measurement_header_mismatch> moe_measure_measurement_header_mismatches(
    const moe_measure_measurement_header & requested,
    const moe_measure_measurement_header & existing,
    bool                            compare_media_configuration = true);
