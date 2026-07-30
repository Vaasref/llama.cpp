#include "moe-measure-common.h"

#include "common.h"
#include "ggml-cpp.h"
#include "gguf.h"
#include "llama.h"

#include <algorithm>
#include <array>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <limits>
#include <random>
#include <regex>
#include <sstream>

namespace {

constexpr uint64_t MOE_MEASURE_FNV_OFFSET           = 14695981039346656037ULL;
constexpr uint64_t MOE_MEASURE_FNV_PRIME            = 1099511628211ULL;
constexpr uint32_t MOE_MEASURE_MEASUREMENT_VERSION  = 5;
constexpr uint64_t MOE_MEASURE_BLOCK_MAGIC          = 0x314b4c424d504552ULL;
constexpr char     MOE_MEASURE_MEASUREMENT_MAGIC[8] = { 'L', 'R', 'E', 'A', 'P', 'M', '1', '\0' };
constexpr uint32_t MOE_MEASURE_MAX_LAYERS            = 512;
constexpr uint32_t MOE_MEASURE_MAX_EXPERTS           = 512;
constexpr uint32_t MOE_MEASURE_MAX_SPLITS            = 1024;
constexpr size_t   MOE_MEASURE_MAX_BLOCK_BYTES       = size_t(1) << 30;

bool checked_mul(size_t a, size_t b, size_t & result) {
    if (a != 0 && b > std::numeric_limits<size_t>::max() / a) {
        return false;
    }
    result = a * b;
    return true;
}

bool measurement_observation_count(uint32_t n_tokens, size_t n_layers, uint32_t n_expert_used, size_t & result) {
    size_t token_layers = 0;
    return checked_mul(n_tokens, n_layers, token_layers) && checked_mul(token_layers, n_expert_used, result);
}

bool measurement_payload_size(
        uint32_t n_tokens, size_t n_layers, uint32_t n_expert_used, size_t & result) {
    size_t n_observations = 0;
    size_t token_bytes    = 0;
    size_t expert_bytes   = 0;
    if (!measurement_observation_count(n_tokens, n_layers, n_expert_used, n_observations) ||
        !checked_mul(n_tokens, sizeof(uint64_t) + sizeof(int32_t), token_bytes) ||
        !checked_mul(n_observations, sizeof(uint32_t) + sizeof(float), expert_bytes) ||
        token_bytes > std::numeric_limits<size_t>::max() - 32 ||
        expert_bytes > std::numeric_limits<size_t>::max() - 32 - token_bytes) {
        return false;
    }
    result = 32 + token_bytes + expert_bytes;
    return true;
}

template <typename T> bool read_value(std::istream & input, T & value) {
    input.read(reinterpret_cast<char *>(&value), sizeof(value));
    return input.good();
}

template <typename T> void append_value(std::vector<uint8_t> & data, const T & value) {
    const uint8_t * src = reinterpret_cast<const uint8_t *>(&value);
    data.insert(data.end(), src, src + sizeof(value));
}

template <typename T> bool take_value(const std::vector<uint8_t> & data, size_t & offset, T & value) {
    if (offset + sizeof(value) > data.size()) {
        return false;
    }
    memcpy(&value, data.data() + offset, sizeof(value));
    offset += sizeof(value);
    return true;
}

bool get_u32(const gguf_context * ctx, const std::string & key, uint32_t & value, bool required, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        if (required) {
            error = "missing GGUF key: " + key;
            return false;
        }
        value = 0;
        return true;
    }
    const gguf_type type = gguf_get_kv_type(ctx, id);
    if (type == GGUF_TYPE_UINT32) {
        value = gguf_get_val_u32(ctx, id);
    } else if (type == GGUF_TYPE_INT32) {
        const int32_t v = gguf_get_val_i32(ctx, id);
        if (v < 0) {
            error = "negative GGUF value for key: " + key;
            return false;
        }
        value = static_cast<uint32_t>(v);
    } else {
        error = "unexpected GGUF type for key: " + key;
        return false;
    }
    return true;
}

bool get_u32_or_array(const gguf_context * ctx, const std::string & key, size_t count,
                      std::vector<uint32_t> & values, bool required, std::string & error) {
    const int64_t id = gguf_find_key(ctx, key.c_str());
    if (id < 0) {
        if (required) {
            error = "missing GGUF key: " + key;
            return false;
        }
        values.assign(count, 0);
        return true;
    }
    if (gguf_get_kv_type(ctx, id) != GGUF_TYPE_ARRAY) {
        uint32_t value = 0;
        if (!get_u32(ctx, key, value, required, error)) {
            return false;
        }
        values.assign(count, value);
        return true;
    }
    if (gguf_get_arr_n(ctx, id) != count) {
        error = "unexpected GGUF array for key: " + key;
        return false;
    }
    const gguf_type array_type = gguf_get_arr_type(ctx, id);
    if (array_type == GGUF_TYPE_UINT32) {
        const uint32_t * data = static_cast<const uint32_t *>(gguf_get_arr_data(ctx, id));
        values.assign(data, data + count);
        return true;
    }
    if (array_type == GGUF_TYPE_INT32) {
        const int32_t * data = static_cast<const int32_t *>(gguf_get_arr_data(ctx, id));
        values.clear();
        values.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            if (data[i] < 0) {
                error = "negative GGUF array value for key: " + key;
                return false;
            }
            values.push_back(static_cast<uint32_t>(data[i]));
        }
        return true;
    }
    error = "unexpected GGUF array type for key: " + key;
    return false;
}

uint64_t measurement_header_size(const moe_measure_measurement_header & header) {
    const uint64_t base_size =
        8 + 4 + 4 + 8 + 8 * 5 + 4 * 7 + header.architecture.size() + 4 * header.moe_layers.size();
    return base_size + 8 * 3 + 4 * 4;
}

bool write_measurement_header(std::ostream & output, const moe_measure_measurement_header & header) {
    output.write(MOE_MEASURE_MEASUREMENT_MAGIC, sizeof(MOE_MEASURE_MEASUREMENT_MAGIC));
    const uint32_t version           = header.format_version;
    const uint32_t n_moe_layer       = static_cast<uint32_t>(header.moe_layers.size());
    const uint32_t architecture_size = static_cast<uint32_t>(header.architecture.size());
    const uint32_t flags             = header.router_weights_normalized ? 1u : 0u;
    const uint32_t reserved          = 0;
    const uint64_t size              = measurement_header_size(header);
    output.write(reinterpret_cast<const char *>(&version), sizeof(version));
    output.write(reinterpret_cast<const char *>(&n_moe_layer), sizeof(n_moe_layer));
    output.write(reinterpret_cast<const char *>(&size), sizeof(size));
    output.write(reinterpret_cast<const char *>(&header.model_signature), sizeof(header.model_signature));
    output.write(reinterpret_cast<const char *>(&header.measurement_id.hi), sizeof(header.measurement_id.hi));
    output.write(reinterpret_cast<const char *>(&header.measurement_id.lo), sizeof(header.measurement_id.lo));
    output.write(reinterpret_cast<const char *>(&header.template_hash), sizeof(header.template_hash));
    output.write(reinterpret_cast<const char *>(&header.tokenization_hash), sizeof(header.tokenization_hash));
    output.write(reinterpret_cast<const char *>(&header.n_layer), sizeof(header.n_layer));
    output.write(reinterpret_cast<const char *>(&header.n_expert), sizeof(header.n_expert));
    output.write(reinterpret_cast<const char *>(&header.n_expert_used), sizeof(header.n_expert_used));
    output.write(reinterpret_cast<const char *>(&header.n_ctx), sizeof(header.n_ctx));
    output.write(reinterpret_cast<const char *>(&flags), sizeof(flags));
    output.write(reinterpret_cast<const char *>(&architecture_size), sizeof(architecture_size));
    output.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));
    const uint32_t scope = static_cast<uint32_t>(header.input_scope);
    const uint32_t role  = static_cast<uint32_t>(header.output_role);
    output.write(reinterpret_cast<const char *>(&header.projector_signature), sizeof(header.projector_signature));
    output.write(reinterpret_cast<const char *>(&header.media_config_hash), sizeof(header.media_config_hash));
    output.write(reinterpret_cast<const char *>(&header.media_pipeline_version), sizeof(header.media_pipeline_version));
    output.write(reinterpret_cast<const char *>(&scope), sizeof(scope));
    output.write(reinterpret_cast<const char *>(&header.vocab_mask_hash), sizeof(header.vocab_mask_hash));
    output.write(reinterpret_cast<const char *>(&role), sizeof(role));
    output.write(reinterpret_cast<const char *>(&reserved), sizeof(reserved));
    output.write(header.architecture.data(), header.architecture.size());
    if (!header.moe_layers.empty()) {
        output.write(reinterpret_cast<const char *>(header.moe_layers.data()),
                     sizeof(int32_t) * header.moe_layers.size());
    }
    return output.good();
}

bool read_measurement_header(std::istream &            input,
                             moe_measure_measurement_header & header,
                             uint64_t &                size,
                             std::string &             error) {
    char magic[8];
    input.read(magic, sizeof(magic));
    if (!input.good() || memcmp(magic, MOE_MEASURE_MEASUREMENT_MAGIC, sizeof(magic)) != 0) {
        error = "not a MoE measurement file";
        return false;
    }
    uint32_t version     = 0;
    uint32_t n_moe_layer = 0;
    uint32_t flags       = 0;
    uint32_t reserved    = 0;
    if (!read_value(input, version) || !read_value(input, n_moe_layer) || !read_value(input, size) ||
        !read_value(input, header.model_signature) || !read_value(input, header.measurement_id.hi) ||
        !read_value(input, header.measurement_id.lo) || !read_value(input, header.template_hash) ||
        !read_value(input, header.tokenization_hash) || !read_value(input, header.n_layer) ||
        !read_value(input, header.n_expert) || !read_value(input, header.n_expert_used) ||
        !read_value(input, header.n_ctx) || !read_value(input, flags)) {
        error = "truncated MoE measurement header";
        return false;
    }
    uint32_t architecture_size = 0;
    if (!read_value(input, architecture_size) || !read_value(input, reserved) || architecture_size > (1u << 20)) {
        error = "invalid MoE measurement architecture field";
        return false;
    }
    if (version != MOE_MEASURE_MEASUREMENT_VERSION) {
        error = "unsupported MoE measurement version; expected version 5";
        return false;
    }
    header.format_version = version;
    uint32_t scope = 0;
    uint32_t role  = 0;
    if (!read_value(input, header.projector_signature) || !read_value(input, header.media_config_hash) ||
        !read_value(input, header.media_pipeline_version) || !read_value(input, scope) ||
        !read_value(input, header.vocab_mask_hash) || !read_value(input, role) || !read_value(input, reserved) ||
        scope < MOE_MEASURE_INPUT_SCOPE_TEXT || scope > MOE_MEASURE_INPUT_SCOPE_ALL || role < MOE_MEASURE_OUTPUT_ROLE_PRIMARY ||
        role > MOE_MEASURE_OUTPUT_ROLE_PREFIX) {
        error = "invalid MoE measurement metadata";
        return false;
    }
    header.input_scope = static_cast<moe_measure_input_scope>(scope);
    header.output_role = static_cast<moe_measure_output_role>(role);
    if (header.n_layer == 0 || header.n_layer > MOE_MEASURE_MAX_LAYERS ||
        n_moe_layer == 0 || n_moe_layer > header.n_layer ||
        header.n_expert == 0 || header.n_expert > MOE_MEASURE_MAX_EXPERTS ||
        header.n_expert_used == 0 || header.n_expert_used > header.n_expert ||
        header.n_ctx == 0 || header.n_ctx > INT32_MAX) {
        error = "invalid MoE measurement dimensions";
        return false;
    }
    header.router_weights_normalized = (flags & 1u) != 0;
    header.architecture.resize(architecture_size);
    input.read(header.architecture.data(), architecture_size);
    header.moe_layers.resize(n_moe_layer);
    input.read(reinterpret_cast<char *>(header.moe_layers.data()), sizeof(int32_t) * n_moe_layer);
    if (!input.good() || header.architecture.empty() || size != measurement_header_size(header)) {
        error = "invalid MoE measurement header size";
        return false;
    }
    std::vector<bool> layers_seen(header.n_layer, false);
    for (int32_t layer : header.moe_layers) {
        if (layer < 0 || static_cast<uint32_t>(layer) >= header.n_layer || layers_seen[layer]) {
            error = "invalid MoE measurement layer list";
            return false;
        }
        layers_seen[layer] = true;
    }
    return true;
}

std::vector<uint8_t> serialize_block(const moe_measure_measurement_header & header, const moe_measure_measurement_block & block) {
    std::vector<uint8_t> data;
    const size_t n_observations = static_cast<size_t>(block.n_tokens) * header.moe_layers.size() * header.n_expert_used;
    data.reserve(32 + block.n_tokens * (sizeof(uint64_t) + sizeof(int32_t)) +
                 n_observations * (sizeof(uint32_t) + sizeof(float)));
    append_value(data, block.source_id);
    append_value(data, block.context_hash);
    append_value(data, block.n_tokens);
    append_value(data, static_cast<uint32_t>(header.moe_layers.size()));
    append_value(data, header.n_expert_used);
    append_value(data, uint32_t(0));
    const uint8_t * hashes = reinterpret_cast<const uint8_t *>(block.token_hashes.data());
    data.insert(data.end(), hashes, hashes + block.token_hashes.size() * sizeof(uint64_t));
    const uint8_t * tokens = reinterpret_cast<const uint8_t *>(block.token_ids.data());
    data.insert(data.end(), tokens, tokens + block.token_ids.size() * sizeof(int32_t));
    const uint8_t * ids = reinterpret_cast<const uint8_t *>(block.expert_ids.data());
    data.insert(data.end(), ids, ids + block.expert_ids.size() * sizeof(uint32_t));
    const uint8_t * values = reinterpret_cast<const uint8_t *>(block.contributions.data());
    data.insert(data.end(), values, values + block.contributions.size() * sizeof(float));
    return data;
}

bool deserialize_block(const moe_measure_measurement_header & header,
                       const std::vector<uint8_t> &    data,
                       moe_measure_measurement_block &        block,
                       std::string &                   error) {
    size_t   offset        = 0;
    uint32_t n_moe_layer   = 0;
    uint32_t n_expert_used = 0;
    uint32_t reserved      = 0;
    if (!take_value(data, offset, block.source_id) || !take_value(data, offset, block.context_hash) ||
        !take_value(data, offset, block.n_tokens) || !take_value(data, offset, n_moe_layer) ||
        !take_value(data, offset, n_expert_used) || !take_value(data, offset, reserved)) {
        error = "invalid MoE measurement block header";
        return false;
    }
    if (block.n_tokens == 0 || n_moe_layer != header.moe_layers.size() || n_expert_used != header.n_expert_used) {
        error = "MoE measurement block dimensions do not match its file header";
        return false;
    }
    size_t n_observations = 0;
    size_t expected       = 0;
    if (block.n_tokens > header.n_ctx ||
        !measurement_observation_count(block.n_tokens, n_moe_layer, n_expert_used, n_observations) ||
        !measurement_payload_size(block.n_tokens, n_moe_layer, n_expert_used, expected) ||
        expected > MOE_MEASURE_MAX_BLOCK_BYTES ||
        expected != data.size()) {
        error = "invalid MoE measurement block payload size";
        return false;
    }
    block.token_hashes.resize(block.n_tokens);
    block.token_ids.resize(block.n_tokens);
    block.expert_ids.resize(n_observations);
    block.contributions.resize(n_observations);
    memcpy(block.token_hashes.data(), data.data() + offset, block.n_tokens * sizeof(uint64_t));
    offset += block.n_tokens * sizeof(uint64_t);
    memcpy(block.token_ids.data(), data.data() + offset, block.n_tokens * sizeof(int32_t));
    offset += block.n_tokens * sizeof(int32_t);
    memcpy(block.expert_ids.data(), data.data() + offset, n_observations * sizeof(uint32_t));
    offset += n_observations * sizeof(uint32_t);
    memcpy(block.contributions.data(), data.data() + offset, n_observations * sizeof(float));
    for (int32_t token : block.token_ids) {
        if (token < -1) {
            error = "MoE measurement block contains an invalid token ID";
            return false;
        }
    }
    for (uint32_t id : block.expert_ids) {
        if (id >= header.n_expert) {
            error = "MoE measurement block contains an invalid expert ID";
            return false;
        }
    }
    return true;
}

struct moe_measure_tensor_descriptor {
    std::string            name;
    std::array<int64_t, 4> ne;
};

void hash_tensor_metadata(uint64_t &                            hash,
                          const gguf_context *                  ctx,
                          ggml_context *                        meta,
                          std::vector<int32_t> *                moe_layers,
                          std::vector<moe_measure_tensor_descriptor> * descriptors = nullptr) {
    static const std::regex expert_pattern(R"(^blk\.(\d+)\.ffn_(gate|up|down|gate_up)_exps\.)");
    const int64_t           n_tensors = gguf_get_n_tensors(ctx);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char *  name   = gguf_get_tensor_name(ctx, i);
        ggml_tensor * tensor = ggml_get_tensor(meta, name);
        hash                 = moe_measure_hash_string(hash, name);
        const uint32_t type  = static_cast<uint32_t>(tensor->type);
        hash                 = moe_measure_hash_bytes(hash, &type, sizeof(type));
        hash                 = moe_measure_hash_bytes(hash, tensor->ne, sizeof(tensor->ne));
        if (descriptors != nullptr) {
            moe_measure_tensor_descriptor descriptor;
            descriptor.name = name;
            std::copy(std::begin(tensor->ne), std::end(tensor->ne), descriptor.ne.begin());
            descriptors->push_back(std::move(descriptor));
        }
        if (moe_layers != nullptr) {
            std::cmatch match;
            if (std::regex_search(name, match, expert_pattern)) {
                int32_t      layer = 0;
                const char * begin = match[1].first;
                const char * end   = match[1].second;
                const auto   parsed = std::from_chars(begin, end, layer);
                if (parsed.ec != std::errc() || parsed.ptr != end ||
                    layer < 0 || layer >= (int32_t) MOE_MEASURE_MAX_LAYERS) {
                    continue;
                }
                if (std::find(moe_layers->begin(), moe_layers->end(), layer) == moe_layers->end()) {
                    moe_layers->push_back(layer);
                }
            }
        }
    }
}


}  // namespace

uint64_t moe_measure_hash_bytes(uint64_t hash, const void * data, size_t size) {
    const uint8_t * bytes = static_cast<const uint8_t *>(data);
    if (hash == 0) {
        hash = MOE_MEASURE_FNV_OFFSET;
    }
    for (size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= MOE_MEASURE_FNV_PRIME;
    }
    return hash;
}

uint64_t moe_measure_hash_string(uint64_t hash, const std::string & value) {
    const uint64_t size = value.size();
    hash                = moe_measure_hash_bytes(hash, &size, sizeof(size));
    return moe_measure_hash_bytes(hash, value.data(), value.size());
}

uint64_t moe_measure_hash_tokens(const std::vector<int32_t> & tokens) {
    return moe_measure_hash_bytes(0, tokens.data(), tokens.size() * sizeof(int32_t));
}

moe_measure_id moe_measure_make_id(const std::string & salt) {
    const auto         now = std::chrono::high_resolution_clock::now().time_since_epoch().count();
    std::random_device random;
    const uint64_t     r0 = (uint64_t(random()) << 32) ^ random();
    const uint64_t     r1 = (uint64_t(random()) << 32) ^ random();
    moe_measure_id            result;
    result.hi = moe_measure_hash_bytes(moe_measure_hash_string(0, salt), &now, sizeof(now));
    result.hi = moe_measure_hash_bytes(result.hi, &r0, sizeof(r0));
    result.lo = moe_measure_hash_bytes(moe_measure_hash_string(0, salt), &r1, sizeof(r1));
    result.lo = moe_measure_hash_bytes(result.lo, &now, sizeof(now));
    return result;
}

std::string moe_measure_id_string(const moe_measure_id & id) {
    std::ostringstream output;
    output << std::hex << std::setfill('0') << std::setw(16) << id.hi << std::setw(16) << id.lo;
    return output.str();
}

bool moe_measure_calculate_contribution(double  router_weight,
                                 double  selected_weight_sum,
                                 double  expert_output_norm_squared,
                                 bool    normalize_router_weight,
                                 float & contribution) {
    if (!std::isfinite(router_weight) || !std::isfinite(selected_weight_sum) ||
        !std::isfinite(expert_output_norm_squared) || router_weight < 0.0 || expert_output_norm_squared < 0.0 ||
        (normalize_router_weight && !(selected_weight_sum > 0.0))) {
        return false;
    }
    const double weight = normalize_router_weight ? router_weight / selected_weight_sum : router_weight;
    const double value  = weight * std::sqrt(expert_output_norm_squared);
    if (!std::isfinite(value) || value < 0.0 || value > std::numeric_limits<float>::max()) {
        return false;
    }
    contribution = static_cast<float>(value);
    return true;
}

bool moe_measure_model_info_load(const std::string & path, moe_measure_model_info & info, std::string & error) {
    info                  = {};
    char prefix[4096]     = { 0 };
    char split_path[4096] = { 0 };
    strncpy(split_path, path.c_str(), sizeof(split_path) - 1);

    ggml_context *   meta0   = nullptr;
    gguf_init_params params0 = { true, &meta0 };
    gguf_context_ptr gguf0(gguf_init_from_file(path.c_str(), params0));
    if (!gguf0) {
        error = "failed to read GGUF metadata from " + path;
        return false;
    }
    const int64_t architecture_key = gguf_find_key(gguf0.get(), "general.architecture");
    if (architecture_key < 0 || gguf_get_kv_type(gguf0.get(), architecture_key) != GGUF_TYPE_STRING) {
        error = "model is missing general.architecture";
        ggml_free(meta0);
        return false;
    }
    info.architecture = gguf_get_val_str(gguf0.get(), architecture_key);
    if (!get_u32(gguf0.get(), info.architecture + ".block_count", info.n_layer, true, error)) {
        ggml_free(meta0);
        return false;
    }
    if (!get_u32(gguf0.get(), info.architecture + ".embedding_length", info.n_embd, true, error)) {
        ggml_free(meta0);
        return false;
    }
    if (info.n_layer == 0 || info.n_layer > MOE_MEASURE_MAX_LAYERS ||
        info.n_embd == 0 || info.n_embd > (1u << 20)) {
        error = "model has invalid layer or embedding dimensions";
        ggml_free(meta0);
        return false;
    }
    if (!get_u32_or_array(gguf0.get(), info.architecture + ".expert_count", info.n_layer,
                          info.n_expert_per_layer, true, error) ||
        !get_u32(gguf0.get(), info.architecture + ".expert_used_count", info.n_expert_used, true, error) ||
        !get_u32(gguf0.get(), info.architecture + ".expert_group_count", info.n_expert_group, false, error) ||
        !get_u32(gguf0.get(), info.architecture + ".expert_group_used_count", info.n_expert_group_used, false, error)) {
        ggml_free(meta0);
        return false;
    }
    if (info.n_expert_per_layer.empty() || info.n_expert_used == 0) {
        error = "model has no routed experts";
        ggml_free(meta0);
        return false;
    }
    info.n_expert = *std::max_element(info.n_expert_per_layer.begin(), info.n_expert_per_layer.end());
    if (info.n_expert == 0 || info.n_expert > MOE_MEASURE_MAX_EXPERTS || info.n_expert_used > info.n_expert) {
        error = "model has invalid expert counts";
        ggml_free(meta0);
        return false;
    }
    uint32_t n_nextn = 0;
    if (!get_u32(gguf0.get(), info.architecture + ".nextn_predict_layers", n_nextn, false, error) ||
        n_nextn >= info.n_layer) {
        error = "model has invalid MTP/NextN layer metadata";
        ggml_free(meta0);
        return false;
    }
    const int64_t split_count_key = gguf_find_key(gguf0.get(), LLM_KV_SPLIT_COUNT);
    if (split_count_key >= 0) {
        if (gguf_get_kv_type(gguf0.get(), split_count_key) != GGUF_TYPE_UINT16) {
            error = "model has an invalid split count type";
            ggml_free(meta0);
            return false;
        }
        info.n_split = gguf_get_val_u16(gguf0.get(), split_count_key);
        if (info.n_split == 0) {
            info.n_split = 1;
        }
        if (info.n_split > MOE_MEASURE_MAX_SPLITS) {
            error = "model has too many splits";
            ggml_free(meta0);
            return false;
        }
    }
    if (info.n_split > 1 && !llama_split_prefix(prefix, sizeof(prefix), path.c_str(), 0, info.n_split)) {
        error = "input is split but its filename is not a valid first-shard name";
        ggml_free(meta0);
        return false;
    }

    uint64_t                            metadata_hash = 0;
    std::vector<moe_measure_tensor_descriptor> tensors;

    for (uint32_t i = 0; i < info.n_split; ++i) {
        std::string current = path;
        if (info.n_split > 1) {
            llama_split_path(split_path, sizeof(split_path), prefix, i, info.n_split);
            current = split_path;
        }
        info.split_paths.push_back(current);
        if (i == 0) {
            hash_tensor_metadata(metadata_hash, gguf0.get(), meta0, &info.moe_layers, &tensors);
            continue;
        }
        ggml_context *   meta   = nullptr;
        gguf_init_params params = { true, &meta };
        gguf_context_ptr gguf(gguf_init_from_file(current.c_str(), params));
        if (!gguf) {
            error = "failed to read split metadata from " + current;
            ggml_free(meta0);
            return false;
        }
        hash_tensor_metadata(metadata_hash, gguf.get(), meta, &info.moe_layers, &tensors);
        ggml_free(meta);
    }
    ggml_free(meta0);
    std::sort(tensors.begin(), tensors.end(), [](const auto & a, const auto & b) { return a.name < b.name; });
    uint64_t structure_hash = moe_measure_hash_string(0, info.architecture);
    structure_hash          = moe_measure_hash_bytes(structure_hash, &info.n_layer, sizeof(info.n_layer));
    structure_hash          = moe_measure_hash_bytes(structure_hash, &info.n_expert, sizeof(info.n_expert));
    structure_hash          = moe_measure_hash_bytes(structure_hash, &info.n_expert_used, sizeof(info.n_expert_used));
    structure_hash          = moe_measure_hash_bytes(structure_hash, &info.n_expert_group, sizeof(info.n_expert_group));
    structure_hash = moe_measure_hash_bytes(structure_hash, &info.n_expert_group_used, sizeof(info.n_expert_group_used));
    for (const moe_measure_tensor_descriptor & tensor : tensors) {
        structure_hash = moe_measure_hash_string(structure_hash, tensor.name);
        structure_hash = moe_measure_hash_bytes(structure_hash, tensor.ne.data(), tensor.ne.size() * sizeof(tensor.ne[0]));
    }
    info.signature = structure_hash;
    std::sort(info.moe_layers.begin(), info.moe_layers.end());
    if (n_nextn > 0) {
        const int32_t main_layer_count = static_cast<int32_t>(info.n_layer - n_nextn);
        info.moe_layers.erase(
            std::lower_bound(info.moe_layers.begin(), info.moe_layers.end(), main_layer_count),
            info.moe_layers.end());
    }
    if (info.moe_layers.empty()) {
        error = "model metadata contains no standard stacked expert tensors";
        return false;
    }
    for (int32_t layer : info.moe_layers) {
        if (layer < 0 || static_cast<uint32_t>(layer) >= info.n_layer || info.n_expert_per_layer[layer] == 0 ||
            info.n_expert_used > info.n_expert_per_layer[layer]) {
            error = "model has invalid per-layer expert counts";
            return false;
        }
    }
    info.n_expert = 0;
    for (int32_t layer : info.moe_layers) {
        info.n_expert = std::max(info.n_expert, info.n_expert_per_layer[layer]);
    }
    return true;
}

bool moe_measure_gguf_structural_signature(const std::string & path, uint64_t & signature, std::string & error) {
    ggml_context *   meta   = nullptr;
    gguf_init_params params = { true, &meta };
    gguf_context_ptr gguf(gguf_init_from_file(path.c_str(), params));
    if (!gguf || meta == nullptr) {
        error = "failed to read GGUF metadata from " + path;
        if (meta != nullptr) {
            ggml_free(meta);
        }
        return false;
    }
    uint64_t      hash             = 0;
    const int64_t architecture_key = gguf_find_key(gguf.get(), "general.architecture");
    if (architecture_key >= 0 && gguf_get_kv_type(gguf.get(), architecture_key) == GGUF_TYPE_STRING) {
        hash = moe_measure_hash_string(hash, gguf_get_val_str(gguf.get(), architecture_key));
    }
    const int64_t type_key = gguf_find_key(gguf.get(), "general.type");
    if (type_key >= 0 && gguf_get_kv_type(gguf.get(), type_key) == GGUF_TYPE_STRING) {
        hash = moe_measure_hash_string(hash, gguf_get_val_str(gguf.get(), type_key));
    }
    hash_tensor_metadata(hash, gguf.get(), meta, nullptr);
    ggml_free(meta);
    signature = hash;
    return true;
}

bool moe_measure_measurement_create(const std::string & path, const moe_measure_measurement_header & header, std::string & error) {
    if (header.format_version != MOE_MEASURE_MEASUREMENT_VERSION || header.architecture.empty() ||
        header.architecture.size() > (1u << 20) ||
        header.n_layer == 0 || header.n_layer > MOE_MEASURE_MAX_LAYERS ||
        header.moe_layers.empty() || header.moe_layers.size() > header.n_layer ||
        header.n_expert == 0 || header.n_expert > MOE_MEASURE_MAX_EXPERTS ||
        header.n_expert_used == 0 || header.n_expert_used > header.n_expert ||
        header.n_ctx == 0 || header.n_ctx > INT32_MAX ||
        header.input_scope < MOE_MEASURE_INPUT_SCOPE_TEXT || header.input_scope > MOE_MEASURE_INPUT_SCOPE_ALL ||
        header.output_role < MOE_MEASURE_OUTPUT_ROLE_PRIMARY || header.output_role > MOE_MEASURE_OUTPUT_ROLE_PREFIX) {
        error = "cannot create a measurement with invalid dimensions";
        return false;
    }
    std::vector<bool> layers_seen(header.n_layer, false);
    for (int32_t layer : header.moe_layers) {
        if (layer < 0 || static_cast<uint32_t>(layer) >= header.n_layer || layers_seen[layer]) {
            error = "cannot create a measurement with an invalid layer list";
            return false;
        }
        layers_seen[layer] = true;
    }
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output || !write_measurement_header(output, header)) {
        error = "failed to write measurement header to " + path;
        return false;
    }
    output.flush();
    return output.good();
}

bool moe_measure_measurement_read(const std::string &                                         path,
                           moe_measure_measurement_summary &                                  summary,
                           const std::function<bool(const moe_measure_measurement_block &)> & callback,
                           std::string &                                               error) {
    summary = {};
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open measurement " + path;
        return false;
    }
    uint64_t header_size = 0;
    if (!read_measurement_header(input, summary.header, header_size, error)) {
        return false;
    }
    summary.valid_size = header_size;
    uint64_t digest    = moe_measure_hash_string(0, summary.header.architecture);
    digest = moe_measure_hash_bytes(digest, &summary.header.model_signature, sizeof(summary.header.model_signature));
    digest = moe_measure_hash_bytes(digest, &summary.header.measurement_id.hi, sizeof(summary.header.measurement_id.hi));
    digest = moe_measure_hash_bytes(digest, &summary.header.measurement_id.lo, sizeof(summary.header.measurement_id.lo));
    digest = moe_measure_hash_bytes(digest, &summary.header.template_hash, sizeof(summary.header.template_hash));
    digest = moe_measure_hash_bytes(digest, &summary.header.tokenization_hash, sizeof(summary.header.tokenization_hash));
    digest = moe_measure_hash_bytes(digest, &summary.header.n_layer, sizeof(summary.header.n_layer));
    digest = moe_measure_hash_bytes(digest, &summary.header.n_expert, sizeof(summary.header.n_expert));
    digest = moe_measure_hash_bytes(digest, &summary.header.n_expert_used, sizeof(summary.header.n_expert_used));
    digest = moe_measure_hash_bytes(digest, &summary.header.n_ctx, sizeof(summary.header.n_ctx));
    const uint32_t metric_flags = summary.header.router_weights_normalized ? 1u : 0u;
    digest                      = moe_measure_hash_bytes(digest, &metric_flags, sizeof(metric_flags));
    digest = moe_measure_hash_bytes(digest, summary.header.moe_layers.data(),
                             summary.header.moe_layers.size() * sizeof(summary.header.moe_layers[0]));
    digest = moe_measure_hash_bytes(digest, &summary.header.format_version, sizeof(summary.header.format_version));
    digest = moe_measure_hash_bytes(digest, &summary.header.projector_signature, sizeof(summary.header.projector_signature));
    digest = moe_measure_hash_bytes(digest, &summary.header.media_config_hash, sizeof(summary.header.media_config_hash));
    digest = moe_measure_hash_bytes(digest, &summary.header.media_pipeline_version,
                             sizeof(summary.header.media_pipeline_version));
    digest = moe_measure_hash_bytes(digest, &summary.header.input_scope, sizeof(summary.header.input_scope));
    digest = moe_measure_hash_bytes(digest, &summary.header.output_role, sizeof(summary.header.output_role));
    digest = moe_measure_hash_bytes(digest, &summary.header.vocab_mask_hash, sizeof(summary.header.vocab_mask_hash));
    std::error_code size_error;
    const uint64_t  file_size = std::filesystem::file_size(path, size_error);
    if (size_error) {
        error = "failed to determine measurement size: " + size_error.message();
        return false;
    }

    while (true) {
        const std::streampos block_start = input.tellg();
        uint64_t             magic       = 0;
        input.read(reinterpret_cast<char *>(&magic), sizeof(magic));
        if (input.eof() && input.gcount() == 0) {
            break;
        }
        if (input.gcount() != sizeof(magic)) {
            summary.has_partial_tail = true;
            break;
        }
        if (magic != MOE_MEASURE_BLOCK_MAGIC) {
            error = "invalid block marker at offset " + std::to_string(static_cast<uint64_t>(block_start));
            return false;
        }
        uint64_t size     = 0;
        uint64_t checksum = 0;
        if (!read_value(input, size) || !read_value(input, checksum)) {
            summary.has_partial_tail = true;
            break;
        }
        size_t maximum_payload = 0;
        const bool has_maximum_payload = measurement_payload_size(
            summary.header.n_ctx, summary.header.moe_layers.size(),
            summary.header.n_expert_used, maximum_payload);
        if (size > MOE_MEASURE_MAX_BLOCK_BYTES ||
            (has_maximum_payload && size > maximum_payload) ||
            size > std::numeric_limits<size_t>::max()) {
            error = "invalid block size at offset " + std::to_string(static_cast<uint64_t>(block_start));
            return false;
        }
        const uint64_t payload_offset = static_cast<uint64_t>(input.tellg());
        if (payload_offset > file_size || size > file_size - payload_offset) {
            summary.has_partial_tail = true;
            break;
        }
        std::vector<uint8_t> payload(size);
        input.read(reinterpret_cast<char *>(payload.data()), size);
        if (!input.good()) {
            error = "failed to read complete block at offset " + std::to_string(static_cast<uint64_t>(block_start));
            return false;
        }
        if (moe_measure_hash_bytes(0, payload.data(), payload.size()) != checksum) {
            error = "block checksum mismatch at offset " + std::to_string(static_cast<uint64_t>(block_start));
            return false;
        }
        moe_measure_measurement_block block;
        if (!deserialize_block(summary.header, payload, block, error)) {
            error = "invalid block at offset " + std::to_string(static_cast<uint64_t>(block_start)) + ": " + error;
            return false;
        }
        summary.context_hashes.insert(block.context_hash);
        summary.has_text_tokens |=
            std::any_of(block.token_ids.begin(), block.token_ids.end(), [](int32_t token) { return token >= 0; });
        summary.has_media_tokens |=
            std::any_of(block.token_ids.begin(), block.token_ids.end(), [](int32_t token) { return token == -1; });
        summary.n_blocks++;
        summary.valid_size = static_cast<uint64_t>(input.tellg());
        digest             = moe_measure_hash_bytes(digest, &checksum, sizeof(checksum));
        if (callback && !callback(block)) {
            error = "measurement callback rejected a block";
            return false;
        }
    }
    summary.content_digest = digest;
    return true;
}

bool moe_measure_measurement_truncate_tail(const std::string &              path,
                                    const moe_measure_measurement_summary & summary,
                                    std::string &                    error) {
    if (!summary.has_partial_tail) {
        return true;
    }
    std::error_code ec;
    std::filesystem::resize_file(path, summary.valid_size, ec);
    if (ec) {
        error = "failed to remove partial measurement tail: " + ec.message();
        return false;
    }
    return true;
}

bool moe_measure_measurement_append(const std::string &             path,
                             const moe_measure_measurement_header & header,
                             const moe_measure_measurement_block &  block,
                             std::string &                   error) {
    size_t n = 0;
    if (block.n_tokens == 0 || block.n_tokens > header.n_ctx ||
        !measurement_observation_count(block.n_tokens, header.moe_layers.size(), header.n_expert_used, n) ||
        block.token_hashes.size() != block.n_tokens || block.token_ids.size() != block.n_tokens ||
        block.expert_ids.size() != n || block.contributions.size() != n) {
        error = "measurement block dimensions are inconsistent";
        return false;
    }
    size_t payload_size = 0;
    if (!measurement_payload_size(
            block.n_tokens, header.moe_layers.size(), header.n_expert_used, payload_size) ||
        payload_size > MOE_MEASURE_MAX_BLOCK_BYTES) {
        error = "measurement block payload is too large";
        return false;
    }
    for (int32_t token : block.token_ids) {
        if (token < -1) {
            error = "measurement block contains an invalid token ID";
            return false;
        }
    }
    const std::vector<uint8_t> payload  = serialize_block(header, block);
    const uint64_t             size     = payload.size();
    const uint64_t             checksum = moe_measure_hash_bytes(0, payload.data(), payload.size());
    std::ofstream              output(path, std::ios::binary | std::ios::app);
    if (!output) {
        error = "failed to open measurement for append: " + path;
        return false;
    }
    output.write(reinterpret_cast<const char *>(&MOE_MEASURE_BLOCK_MAGIC), sizeof(MOE_MEASURE_BLOCK_MAGIC));
    output.write(reinterpret_cast<const char *>(&size), sizeof(size));
    output.write(reinterpret_cast<const char *>(&checksum), sizeof(checksum));
    output.write(reinterpret_cast<const char *>(payload.data()), payload.size());
    output.flush();
    if (!output.good()) {
        error = "failed to append a complete measurement block";
        return false;
    }
    return true;
}

std::vector<moe_measure_measurement_header_mismatch> moe_measure_measurement_header_mismatches(
    const moe_measure_measurement_header & requested,
    const moe_measure_measurement_header & existing,
    bool                            compare_media_configuration) {
    std::vector<moe_measure_measurement_header_mismatch> result;
    const auto add = [&](const char * field, const std::string & requested_value, const std::string & existing_value) {
        if (requested_value != existing_value) {
            result.push_back({ field, requested_value, existing_value });
        }
    };
    const auto decimal = [](auto value) { return std::to_string(value); };
    const auto hex = [](uint64_t value) {
        std::ostringstream output;
        output << "0x" << std::hex << std::setfill('0') << std::setw(16) << value;
        return output.str();
    };
    const auto layers = [](const std::vector<int32_t> & values) {
        std::ostringstream output;
        output << '[';
        for (size_t i = 0; i < values.size(); ++i) {
            if (i > 0) {
                output << ',';
            }
            output << values[i];
        }
        output << ']';
        return output.str();
    };
    const auto input_scope = [](moe_measure_input_scope value) {
        switch (value) {
            case MOE_MEASURE_INPUT_SCOPE_TEXT:  return std::string("text");
            case MOE_MEASURE_INPUT_SCOPE_MEDIA: return std::string("media");
            case MOE_MEASURE_INPUT_SCOPE_ALL:   return std::string("all");
            default:                     return "unknown (" + std::to_string(static_cast<uint32_t>(value)) + ')';
        }
    };
    const auto output_role = [](moe_measure_output_role value) {
        switch (value) {
            case MOE_MEASURE_OUTPUT_ROLE_PRIMARY: return std::string("primary");
            case MOE_MEASURE_OUTPUT_ROLE_PREFIX:  return std::string("prefix");
            default:                       return "unknown (" + std::to_string(static_cast<uint32_t>(value)) + ')';
        }
    };

    add("format_version", decimal(requested.format_version), decimal(existing.format_version));
    add("architecture", requested.architecture, existing.architecture);
    add("model_signature", hex(requested.model_signature), hex(existing.model_signature));
    add("template_hash", hex(requested.template_hash), hex(existing.template_hash));
    add("tokenization_hash", hex(requested.tokenization_hash), hex(existing.tokenization_hash));
    add("n_layer", decimal(requested.n_layer), decimal(existing.n_layer));
    add("n_expert", decimal(requested.n_expert), decimal(existing.n_expert));
    add("n_expert_used", decimal(requested.n_expert_used), decimal(existing.n_expert_used));
    add("n_ctx", decimal(requested.n_ctx), decimal(existing.n_ctx));
    add("router_weights_normalized", requested.router_weights_normalized ? "true" : "false",
        existing.router_weights_normalized ? "true" : "false");
    if (compare_media_configuration) {
        add("projector_signature", hex(requested.projector_signature), hex(existing.projector_signature));
        add("media_config_hash", hex(requested.media_config_hash), hex(existing.media_config_hash));
        add("media_pipeline_version", decimal(requested.media_pipeline_version),
            decimal(existing.media_pipeline_version));
        add("input_scope", input_scope(requested.input_scope), input_scope(existing.input_scope));
    }
    add("output_role", output_role(requested.output_role), output_role(existing.output_role));
    add("vocab_mask_hash", hex(requested.vocab_mask_hash), hex(existing.vocab_mask_hash));
    add("moe_layers", layers(requested.moe_layers), layers(existing.moe_layers));
    return result;
}
