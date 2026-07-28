#include "arg.h"
#include "chat.h"
#include "common.h"
#include "ggml-backend.h"
#include "llama.h"
#include "log.h"
#include "mtmd-helper.h"
#include "mtmd.h"
#include "moe-measure-common.h"
#include "moe-measure-scheduler.h"
#include "moe-measure-media.h"

#include <algorithm>
#include <chrono>
#include <clocale>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <functional>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <nlohmann/json.hpp>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

using json = nlohmann::ordered_json;

namespace {

struct measure_options {
    enum class collector_mode {
        device,
        cpu,
    };

    enum class dataset_kind {
        text,
        chat,
        input_jsonl,
    };

    struct dataset {
        dataset_kind kind;
        std::string  path;
    };

    std::vector<std::string> text_files;
    std::vector<std::string> chat_files;
    std::vector<std::string> input_jsonl_files;
    std::vector<dataset>     datasets;
    std::string              prefix_output;
    bool                     prefix_output_set = false;
    std::vector<std::string> measured_vocab_files;
    std::vector<std::string> excluded_vocab_files;
    bool                     exclude_special_tokens   = false;
    bool                     normalize_router_weights = true;
    std::string              chat_template;
    std::string              chat_template_file;
    std::string              jinja_template_file;
    std::string              chat_template_kwargs;
    int                      use_jinja        = -1;
    int                      enable_reasoning = -2;
    std::string              mmproj;
    bool                     mmproj_offload        = true;
    int                      image_min_tokens      = -1;
    int                      image_max_tokens      = -1;
    int                      mtmd_batch_max_tokens = 1024;
    double                   soft_token_buffer_gib = 0.0;
    bool                     soft_token_buffer_set = false;
    std::string              media_path;
    size_t                   media_max_bytes  = 64u * 1024u * 1024u;
    moe_measure_input_scope         multimodal_scope = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    collector_mode           collector        = collector_mode::device;
};

void print_usage(const char * argv0) {
    LOG("usage: %s -m MODEL -o DATA.moem [--text FILE ...] [--chat FILE ...] [--input-jsonl FILE ...] "
        "[model options]\n",
        argv0);
    LOG("\n");
    LOG("  --text FILE                 append a plain-text dataset (repeatable)\n");
    LOG("  --chat FILE                 append an OpenAI-compatible JSONL dataset (repeatable)\n");
    LOG("  --input-jsonl FILE          append prefix/input JSONL records (repeatable)\n");
    LOG("  --prefix-output FILE        write retained context_prefix observations\n");
    LOG("  --measured-vocab FILE       allow token IDs listed in FILE (repeatable)\n");
    LOG("  --excluded-vocab FILE       deny token IDs listed in FILE (repeatable)\n");
    LOG("  --exclude-special-tokens    deny non-text vocabulary tokens\n");
    LOG("  -np, --parallel N           parallel measurement slots (default 1)\n");
    LOG("  -c, --ctx-size N            context tokens per measurement slot (default 512)\n");
    LOG("  -b, --batch-size N          logical token limit shared by active slots\n");
    LOG("  -ub, --ubatch-size N        physical token limit shared by active slots\n");
    LOG("  --router-weights MODE       renormalized (default) or model\n");
    LOG("  --chat-template TEMPLATE    override the model's embedded template\n");
    LOG("  --chat-template-file FILE   read the template override from a file\n");
    LOG("  --jinja-template FILE       read a Jinja2 template from a file and enable Jinja\n");
    LOG("  --chat-template-kwargs JSON default Jinja template arguments\n");
    LOG("  --reasoning MODE            on, off, or auto (default)\n");
    LOG("  -mm, --mmproj FILE          multimodal projector for image records\n");
    LOG("  --mmproj-offload            enable projector GPU offload (default)\n");
    LOG("  --no-mmproj-offload         disable projector GPU offload\n");
    LOG("  --image-min-tokens N        minimum dynamic-resolution image tokens\n");
    LOG("  --image-max-tokens N        maximum dynamic-resolution image tokens\n");
    LOG("  --mtmd-batch-max-tokens N   projector batch token limit (default 1024)\n");
    LOG("  --soft-token-buffer-gib N   F32 projector buffer in GiB (default 10%% of physical RAM)\n");
    LOG("  --media-path DIR            sandbox root for file://relative images\n");
    LOG("  --media-max-bytes N         maximum encoded bytes per image (default 67108864)\n");
    LOG("  --multimodal-scope SCOPE    media (default), text, or all\n");
    LOG("  --collector-mode MODE       device (default) or cpu\n");
    LOG("\n");
}

bool parse_custom_args(int                   argc,
                       char **               argv,
                       measure_options &     options,
                       std::vector<char *> & common_argv,
                       std::string &         error) {
    bool common_immediate_exit = false;
    common_argv.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            exit(0);
        }
        if (arg == "--jinja" || arg == "--no-jinja") {
            options.use_jinja = arg == "--jinja";
            continue;
        }
        if (arg == "--mmproj-offload" || arg == "--no-mmproj-offload") {
            options.mmproj_offload = arg == "--mmproj-offload";
            continue;
        }
        if (arg == "--exclude-special-tokens") {
            options.exclude_special_tokens = true;
            continue;
        }
        if (arg == "--text" || arg == "--chat" || arg == "--input-jsonl" || arg == "--prefix-output" ||
            arg == "--measured-vocab" || arg == "--excluded-vocab" || arg == "--router-weights" ||
            arg == "--chat-template" || arg == "--chat-template-file" || arg == "--jinja-template" ||
            arg == "--chat-template-kwargs" || arg == "--reasoning" || arg == "-mm" || arg == "--mmproj" ||
            arg == "--image-min-tokens" || arg == "--image-max-tokens" || arg == "--mtmd-batch-max-tokens" ||
            arg == "--soft-token-buffer-gib" || arg == "--media-path" || arg == "--media-max-bytes" ||
            arg == "--multimodal-scope" || arg == "--collector-mode") {
            if (++i >= argc) {
                error = "missing value for " + arg;
                return false;
            }
            if (arg == "--text") {
                options.text_files.push_back(argv[i]);
                options.datasets.push_back({ measure_options::dataset_kind::text, argv[i] });
            } else if (arg == "--chat") {
                options.chat_files.push_back(argv[i]);
                options.datasets.push_back({ measure_options::dataset_kind::chat, argv[i] });
            } else if (arg == "--input-jsonl") {
                options.input_jsonl_files.push_back(argv[i]);
                options.datasets.push_back({ measure_options::dataset_kind::input_jsonl, argv[i] });
            } else if (arg == "--prefix-output") {
                options.prefix_output     = argv[i];
                options.prefix_output_set = true;
            } else if (arg == "--measured-vocab") {
                options.measured_vocab_files.push_back(argv[i]);
            } else if (arg == "--excluded-vocab") {
                options.excluded_vocab_files.push_back(argv[i]);
            } else if (arg == "--router-weights") {
                const std::string mode = argv[i];
                if (mode == "renormalized") {
                    options.normalize_router_weights = true;
                } else if (mode == "model") {
                    options.normalize_router_weights = false;
                } else {
                    error = "--router-weights must be 'renormalized' or 'model'";
                    return false;
                }
            } else if (arg == "--chat-template") {
                options.chat_template = argv[i];
            } else if (arg == "--chat-template-file") {
                options.chat_template_file = argv[i];
            } else if (arg == "--jinja-template") {
                options.jinja_template_file = argv[i];
            } else if (arg == "--chat-template-kwargs") {
                options.chat_template_kwargs = argv[i];
            } else if (arg == "--reasoning") {
                const std::string mode = argv[i];
                if (mode == "on") {
                    options.enable_reasoning = 1;
                } else if (mode == "off") {
                    options.enable_reasoning = 0;
                } else if (mode == "auto") {
                    options.enable_reasoning = -1;
                } else {
                    error = "--reasoning must be on, off, or auto";
                    return false;
                }
            } else if (arg == "-mm" || arg == "--mmproj") {
                options.mmproj = argv[i];
            } else if (arg == "--media-path") {
                options.media_path = argv[i];
            } else if (arg == "--multimodal-scope") {
                const std::string scope = argv[i];
                if (scope == "media") {
                    options.multimodal_scope = MOE_MEASURE_INPUT_SCOPE_MEDIA;
                } else if (scope == "text") {
                    options.multimodal_scope = MOE_MEASURE_INPUT_SCOPE_TEXT;
                } else if (scope == "all") {
                    options.multimodal_scope = MOE_MEASURE_INPUT_SCOPE_ALL;
                } else {
                    error = "--multimodal-scope must be media, text, or all";
                    return false;
                }
            } else if (arg == "--collector-mode") {
                const std::string mode = argv[i];
                if (mode == "device") {
                    options.collector = measure_options::collector_mode::device;
                } else if (mode == "cpu") {
                    options.collector = measure_options::collector_mode::cpu;
                } else {
                    error = "--collector-mode must be device or cpu";
                    return false;
                }
            } else if (arg == "--soft-token-buffer-gib") {
                try {
                    size_t                parsed        = 0;
                    const double          value         = std::stod(argv[i], &parsed);
                    constexpr long double bytes_per_gib = 1024.0L * 1024.0L * 1024.0L;
                    const long double     bytes         = static_cast<long double>(value) * bytes_per_gib;
                    if (parsed != strlen(argv[i]) || !std::isfinite(value) || value <= 0.0 || bytes < 1.0L ||
                        bytes > std::numeric_limits<size_t>::max()) {
                        throw std::invalid_argument("invalid");
                    }
                    options.soft_token_buffer_gib = value;
                    options.soft_token_buffer_set = true;
                } catch (...) {
                    error = "invalid numeric value for " + arg;
                    return false;
                }
            } else {
                try {
                    const long long value = std::stoll(argv[i]);
                    if (arg == "--image-min-tokens") {
                        options.image_min_tokens = static_cast<int>(value);
                    } else if (arg == "--image-max-tokens") {
                        options.image_max_tokens = static_cast<int>(value);
                    } else if (arg == "--mtmd-batch-max-tokens") {
                        options.mtmd_batch_max_tokens = static_cast<int>(value);
                    } else {
                        if (value <= 0) {
                            throw std::out_of_range("non-positive");
                        }
                        options.media_max_bytes = static_cast<size_t>(value);
                    }
                } catch (...) {
                    error = "invalid numeric value for " + arg;
                    return false;
                }
            }
            continue;
        }
        if (arg == "--list-devices" || arg == "--version" || arg == "--completion-bash") {
            common_immediate_exit = true;
        }
        common_argv.push_back(argv[i]);
    }
    if (!common_immediate_exit &&
        options.text_files.empty() && options.chat_files.empty() && options.input_jsonl_files.empty()) {
        error = "at least one --text, --chat, or --input-jsonl dataset is required";
        return false;
    }
    if (options.prefix_output_set && options.prefix_output.empty()) {
        error = "--prefix-output path must not be empty";
        return false;
    }
    if (options.prefix_output_set && options.input_jsonl_files.empty()) {
        error = "--prefix-output requires at least one --input-jsonl dataset";
        return false;
    }
    if (options.mtmd_batch_max_tokens <= 0 || (options.image_min_tokens >= 0 && options.image_max_tokens >= 0 &&
                                               options.image_min_tokens > options.image_max_tokens)) {
        error = "invalid multimodal token limits";
        return false;
    }
    return true;
}

std::string tensor_base_name(const char * name) {
    const char * begin = strchr(name, '#');
    if (begin == nullptr) {
        return name;
    }
    begin++;
    const char * end = strchr(begin, '#');
    return end == nullptr ? std::string(begin) : std::string(begin, end - begin);
}

using moe_measure_clock = std::chrono::steady_clock;

// Retained so version-5 logs produced by the earlier tool have the same resume identity.
constexpr char LEGACY_VOCAB_MASK_SALT[] = "reap-vocab-mask-v1";

double elapsed_ms(moe_measure_clock::time_point start) {
    return std::chrono::duration<double, std::milli>(moe_measure_clock::now() - start).count();
}

struct moe_measure_performance {
    uint64_t callback_calls  = 0;
    uint64_t callback_bytes  = 0;
    uint64_t decoded_tokens  = 0;
    uint64_t measured_tokens = 0;
    double   copy_ms         = 0.0;
    double   scoring_ms      = 0.0;
    double   decode_ms       = 0.0;
    double   preparation_ms  = 0.0;
    double   projector_ms    = 0.0;
    double   commit_ms       = 0.0;
};

struct tensor_snapshot {
    const ggml_tensor * tensor = nullptr;
    const uint8_t *     data   = nullptr;

    template <typename T>
    tensor_snapshot(const ggml_tensor * value, std::vector<T> & storage, moe_measure_performance & perf) : tensor(value) {
        if (ggml_backend_buffer_is_host(value->buffer)) {
            data = static_cast<const uint8_t *>(value->data);
        } else {
            const auto   start = moe_measure_clock::now();
            const size_t bytes = ggml_nbytes(value);
            storage.resize((bytes + sizeof(T) - 1) / sizeof(T));
            ggml_backend_tensor_get(value, storage.data(), 0, bytes);
            data = reinterpret_cast<const uint8_t *>(storage.data());
            perf.copy_ms += elapsed_ms(start);
            perf.callback_bytes += bytes;
        }
    }

    template <typename T> T at(int64_t i0, int64_t i1 = 0, int64_t i2 = 0, int64_t i3 = 0) const {
        const size_t offset = i0 * tensor->nb[0] + i1 * tensor->nb[1] + i2 * tensor->nb[2] + i3 * tensor->nb[3];
        T            value;
        memcpy(&value, data + offset, sizeof(value));
        return value;
    }

    template <typename T> T at_row(int64_t i0, int64_t i1, size_t row) const {
        const int64_t i2 = row % tensor->ne[2];
        const int64_t i3 = row / tensor->ne[2];
        return at<T>(i0, i1, i2, i3);
    }

    template <typename T> T at_after_axis0(int64_t i0, size_t row) const {
        const int64_t i1 = row % tensor->ne[1];
        row /= tensor->ne[1];
        const int64_t i2 = row % tensor->ne[2];
        const int64_t i3 = row / tensor->ne[2];
        return at<T>(i0, i1, i2, i3);
    }
};

struct moe_measure_collect_target {
    moe_measure_measurement_block * block = nullptr;
    size_t                   token = 0;
};

struct moe_measure_collector {
    moe_measure_measurement_header             header;
    std::vector<moe_measure_collect_target>    targets;
    std::vector<moe_measure_collect_target>    submitted_targets;
    std::vector<uint8_t>                       rows_seen;
    std::vector<bool>                   layers_seen;
    std::vector<size_t>                 layer_offsets;
    std::vector<uint32_t>               expert_counts;
    std::unordered_map<int32_t, size_t> layer_to_index;
    measure_options::collector_mode     mode = measure_options::collector_mode::device;
    std::vector<float>                  expert_storage;
    std::vector<std::vector<float>>     cpu_norm2;
    moe_measure_performance                    perf;
    std::string                         error;
    std::mutex                          mutex;

    void init(const moe_measure_measurement_header & value,
              const std::vector<uint32_t> &   per_layer_experts,
              measure_options::collector_mode collector_mode) {
        header        = value;
        expert_counts = per_layer_experts;
        mode          = collector_mode;
        for (size_t i = 0; i < header.moe_layers.size(); ++i) {
            layer_to_index[header.moe_layers[i]] = i;
        }
    }

    void begin(std::vector<moe_measure_collect_target> value) {
        std::lock_guard<std::mutex> lock(mutex);
        submitted_targets = std::move(value);
        targets.clear();
        targets.reserve(submitted_targets.size());
        rows_seen.assign(submitted_targets.size(), 0);
        perf.measured_tokens += std::count_if(
            submitted_targets.begin(), submitted_targets.end(),
            [](const moe_measure_collect_target & target) { return target.block != nullptr; });
        layers_seen.assign(header.moe_layers.size(), false);
        layer_offsets.assign(header.moe_layers.size(), 0);
        cpu_norm2.assign(header.moe_layers.size(), {});
        error.clear();
    }

    void disable() {
        std::lock_guard<std::mutex> lock(mutex);
        targets.clear();
        submitted_targets.clear();
        rows_seen.clear();
        layers_seen.clear();
        layer_offsets.clear();
        cpu_norm2.clear();
        error.clear();
    }

    void reorder(const int32_t * row_ids, size_t n_rows) {
        std::lock_guard<std::mutex> lock(mutex);
        if (submitted_targets.empty() || !error.empty()) {
            return;
        }
        if (!moe_measure_append_row_order(row_ids, n_rows, submitted_targets.size(), rows_seen)) {
            error = "invalid internal decode row order";
            return;
        }
        for (size_t i = 0; i < n_rows; ++i) {
            targets.push_back(submitted_targets[row_ids[i]]);
        }
    }

    bool finish() {
        std::lock_guard<std::mutex> lock(mutex);
        if (!error.empty()) {
            return false;
        }
        if (targets.size() != submitted_targets.size()) {
            error = "incomplete internal decode row order";
            return false;
        }
        for (size_t i = 0; i < layers_seen.size(); ++i) {
            if (!layers_seen[i]) {
                error = "no MoE expert capture callback for layer " + std::to_string(header.moe_layers[i]);
                return false;
            }
            if (layer_offsets[i] != targets.size()) {
                error = "incomplete MoE expert capture rows for layer " + std::to_string(header.moe_layers[i]);
                return false;
            }
        }
        targets.clear();
        return true;
    }

    bool collect(ggml_tensor * tensor, bool ask) {
        const std::string name = tensor_base_name(tensor->name);
        const std::string prefix = "ffn_moe_expert_capture-";
        const std::string expert_output_prefix = "ffn_moe_expert_output-";
        const bool is_cpu_auxiliary =
            mode == measure_options::collector_mode::cpu &&
            name.compare(0, expert_output_prefix.size(), expert_output_prefix) == 0;
        if (name.compare(0, prefix.size(), prefix) != 0 && !is_cpu_auxiliary) {
            return false;
        }
        std::lock_guard<std::mutex> lock(mutex);
        if (ask) {
            return !targets.empty();
        }
        if (targets.empty() || !error.empty()) {
            return true;
        }
        if (is_cpu_auxiliary) {
            int32_t layer = -1;
            try {
                layer = std::stoi(name.substr(expert_output_prefix.size()));
            } catch (const std::exception &) {
                error = "cannot parse MoE layer from callback tensor " + name;
                return true;
            }
            const auto layer_it = layer_to_index.find(layer);
            if (layer_it == layer_to_index.end()) {
                error = "callback reported unexpected MoE layer " + std::to_string(layer);
                return true;
            }
            const size_t layer_index = layer_it->second;
            if (tensor->type != GGML_TYPE_F32 || tensor->ne[1] != header.n_expert_used) {
                error = "unexpected unweighted expert output dimensions in layer " + std::to_string(layer);
                return true;
            }
            const size_t n_rows = ggml_nelements(tensor) / (tensor->ne[0] * tensor->ne[1]);
            tensor_snapshot data(tensor, expert_storage, perf);
            cpu_norm2[layer_index].assign(n_rows * header.n_expert_used, 0.0f);
            for (size_t row = 0; row < n_rows; ++row) {
                for (size_t k = 0; k < header.n_expert_used; ++k) {
                    double value = 0.0;
                    for (int64_t j = 0; j < tensor->ne[0]; ++j) {
                        const float element = data.at_row<float>(j, k, row);
                        value += double(element) * element;
                    }
                    cpu_norm2[layer_index][row * header.n_expert_used + k] = static_cast<float>(value);
                }
            }
            return true;
        }
        int32_t layer = -1;
        try {
            layer = std::stoi(name.substr(prefix.size()));
        } catch (const std::exception &) {
            error = "cannot parse MoE layer from callback tensor " + name;
            return true;
        }
        const auto layer_it = layer_to_index.find(layer);
        if (layer_it == layer_to_index.end()) {
            error = "callback reported unexpected MoE layer " + std::to_string(layer);
            return true;
        }
        const size_t layer_index = layer_it->second;
        const ggml_tensor * experts = tensor;
        size_t              n_rows  = 0;
        if (tensor->type != GGML_TYPE_F32 || tensor->ne[0] != 3 ||
            tensor->ne[1] != header.n_expert_used) {
            error = "unexpected MoE measurement capture dimensions in layer " + std::to_string(layer) + ": capture=[" +
                    std::to_string(tensor->ne[0]) + "," + std::to_string(tensor->ne[1]) + "," +
                    std::to_string(tensor->ne[2]) + "," + std::to_string(tensor->ne[3]) + "]";
            return true;
        }
        n_rows  = ggml_nelements(tensor) / (tensor->ne[0] * tensor->ne[1]);
        if (mode == measure_options::collector_mode::cpu &&
            cpu_norm2[layer_index].size() != n_rows * header.n_expert_used) {
            error = "incomplete CPU collector inputs in layer " + std::to_string(layer);
            return true;
        }
        if (n_rows == 0) {
            return true;
        }
        if (layer_offsets[layer_index] > targets.size() ||
            n_rows > targets.size() - layer_offsets[layer_index]) {
            const size_t remaining =
                layer_offsets[layer_index] <= targets.size() ? targets.size() - layer_offsets[layer_index] : 0;
            error = "callback supplied " + std::to_string(n_rows) + " rows for MoE layer " +
                    std::to_string(layer) + " with " + std::to_string(remaining) + " batch rows remaining";
            return true;
        }

        tensor_snapshot expert_data(experts, expert_storage, perf);
        perf.callback_calls++;
        const auto scoring_start = moe_measure_clock::now();
        size_t     target_offset = 0;
        if (!moe_measure_consume_layer_rows(layer_offsets, layer_index, n_rows, targets.size(), target_offset)) {
            error = "callback supplied too many rows for MoE layer " + std::to_string(layer);
            return true;
        }
        for (size_t token = 0; token < n_rows; ++token) {
            const moe_measure_collect_target target = targets[target_offset + token];
            if (target.block == nullptr) {
                continue;
            }
            double weight_sum = 0.0;
            for (size_t k = 0; k < header.n_expert_used; ++k) {
                weight_sum += expert_data.at_row<float>(1, k, token);
            }
            if (header.router_weights_normalized && !(weight_sum > 0.0)) {
                error = "router weights do not have a positive sum in layer " + std::to_string(layer);
                return true;
            }
            for (size_t k = 0; k < header.n_expert_used; ++k) {
                int32_t expert = 0;
                const float encoded_expert = expert_data.at_row<float>(2, k, token);
                if (!moe_measure_decode_captured_expert(encoded_expert, expert_counts[layer_index], expert)) {
                    error = "selected expert ID " + std::to_string(encoded_expert) +
                            " is not an integer in range [0," + std::to_string(expert_counts[layer_index]) +
                            ") in layer " + std::to_string(layer) + ", batch row " + std::to_string(token) +
                            ", route " + std::to_string(k);
                    return true;
                }
                if (expert < 0 || expert >= (int32_t) expert_counts[layer_index]) {
                    error = "selected expert ID " + std::to_string(expert) + " is out of range [0," +
                            std::to_string(expert_counts[layer_index]) + ") in layer " + std::to_string(layer) +
                            ", batch row " + std::to_string(token) + ", route " + std::to_string(k);
                    return true;
                }
                double norm2 =
                    mode == measure_options::collector_mode::device ?
                        expert_data.at_row<float>(0, k, token) :
                        cpu_norm2[layer_index][token * header.n_expert_used + k];
                const size_t output =
                    (target.token * header.moe_layers.size() + layer_index) * header.n_expert_used + k;
                if (target.token >= target.block->n_tokens || output >= target.block->expert_ids.size() ||
                    output >= target.block->contributions.size()) {
                    error = "collector observation index is out of range in layer " + std::to_string(layer) +
                            ": token " + std::to_string(target.token) + " of " +
                            std::to_string(target.block->n_tokens);
                    return true;
                }
                float contribution = 0.0f;
                const float router_weight = expert_data.at_row<float>(1, k, token);
                if (!moe_measure_calculate_contribution(router_weight, weight_sum, norm2,
                                                 header.router_weights_normalized, contribution)) {
                    error = "non-finite or negative MoE measurement contribution in layer " + std::to_string(layer);
                    return true;
                }
                target.block->expert_ids[output]    = expert;
                target.block->contributions[output] = contribution;
            }
        }
        perf.scoring_ms += elapsed_ms(scoring_start);
        if (mode == measure_options::collector_mode::cpu) {
            cpu_norm2[layer_index].clear();
        }
        layers_seen[layer_index] = true;
        return true;
    }
};

bool collector_callback(ggml_tensor * tensor, bool ask, void * user_data) {
    return static_cast<moe_measure_collector *>(user_data)->collect(tensor, ask);
}

void collector_row_order_callback(const int32_t * row_ids, size_t n_rows, void * user_data) {
    static_cast<moe_measure_collector *>(user_data)->reorder(row_ids, n_rows);
}

uint64_t hash_vocab(const llama_vocab * vocab) {
    uint64_t      hash  = 0;
    const int32_t count = llama_vocab_n_tokens(vocab);
    hash                = moe_measure_hash_bytes(hash, &count, sizeof(count));
    for (llama_token token = 0; token < count; ++token) {
        const char * text            = llama_vocab_get_text(vocab, token);
        hash                         = moe_measure_hash_string(hash, text == nullptr ? std::string() : std::string(text));
        const float            score = llama_vocab_get_score(vocab, token);
        const llama_token_attr attr  = llama_vocab_get_attr(vocab, token);
        hash                         = moe_measure_hash_bytes(hash, &score, sizeof(score));
        hash                         = moe_measure_hash_bytes(hash, &attr, sizeof(attr));
    }
    return hash;
}

bool read_file(const std::string & path, std::string & data, std::string & error, const char * kind = "dataset") {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "failed to open " + std::string(kind) + " " + path;
        return false;
    }
    std::ostringstream buffer;
    buffer << input.rdbuf();
    if (!input.good() && !input.eof()) {
        error = "failed to read " + std::string(kind) + " " + path;
        return false;
    }
    data = buffer.str();
    return true;
}

struct moe_measure_vocab_mask {
    std::vector<uint8_t> selected;
    uint64_t             hash = 0;

    bool includes(llama_token token) const {
        return token >= 0 && static_cast<size_t>(token) < selected.size() && selected[token] != 0;
    }
};

bool read_vocab_ids(const std::string & path, size_t n_vocab, std::vector<uint8_t> & ids, std::string & error) {
    std::ifstream input(path);
    if (!input) {
        error = "failed to open vocabulary token-ID file " + path;
        return false;
    }
    std::string line;
    size_t      line_number = 0;
    while (std::getline(input, line)) {
        line_number++;
        const size_t comment = line.find('#');
        if (comment != std::string::npos) {
            line.resize(comment);
        }
        const size_t begin = line.find_first_not_of(" \t\r\n");
        if (begin == std::string::npos) {
            continue;
        }
        const size_t      end  = line.find_last_not_of(" \t\r\n");
        const std::string text = line.substr(begin, end - begin + 1);
        try {
            size_t          parsed = 0;
            const long long value  = std::stoll(text, &parsed, 10);
            if (parsed != text.size() || value < 0 || static_cast<uint64_t>(value) >= n_vocab) {
                throw std::out_of_range("token ID");
            }
            ids[static_cast<size_t>(value)] = 1;
        } catch (...) {
            error = path + ":" + std::to_string(line_number) + ": invalid or out-of-range token ID";
            return false;
        }
    }
    if (!input.eof()) {
        error = "failed to read vocabulary token-ID file " + path;
        return false;
    }
    return true;
}

bool build_vocab_mask(const measure_options & options,
                      const llama_vocab *     vocab,
                      moe_measure_vocab_mask &       mask,
                      std::string &           error) {
    const size_t n_vocab       = llama_vocab_n_tokens(vocab);
    const bool   explicit_mask = !options.measured_vocab_files.empty() || !options.excluded_vocab_files.empty() ||
                                 options.exclude_special_tokens;
    mask.selected.assign(n_vocab, options.measured_vocab_files.empty() ? 1 : 0);
    std::vector<uint8_t> ids(n_vocab, 0);
    for (const std::string & path : options.measured_vocab_files) {
        std::fill(ids.begin(), ids.end(), 0);
        if (!read_vocab_ids(path, n_vocab, ids, error)) {
            return false;
        }
        for (size_t i = 0; i < n_vocab; ++i) {
            mask.selected[i] |= ids[i];
        }
    }
    for (const std::string & path : options.excluded_vocab_files) {
        std::fill(ids.begin(), ids.end(), 0);
        if (!read_vocab_ids(path, n_vocab, ids, error)) {
            return false;
        }
        for (size_t i = 0; i < n_vocab; ++i) {
            if (ids[i]) {
                mask.selected[i] = 0;
            }
        }
    }
    if (options.exclude_special_tokens) {
        constexpr uint32_t special_attrs = LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_USER_DEFINED |
                                           LLAMA_TOKEN_ATTR_UNKNOWN | LLAMA_TOKEN_ATTR_UNUSED;
        for (size_t i = 0; i < n_vocab; ++i) {
            const llama_token token = static_cast<llama_token>(i);
            if ((llama_vocab_get_attr(vocab, token) & special_attrs) != 0 || llama_vocab_is_eog(vocab, token)) {
                mask.selected[i] = 0;
            }
        }
    }
    if (std::none_of(mask.selected.begin(), mask.selected.end(), [](uint8_t value) { return value != 0; })) {
        error = "vocabulary filters exclude every token";
        return false;
    }
    mask.hash = explicit_mask ? moe_measure_hash_bytes(moe_measure_hash_string(0, LEGACY_VOCAB_MASK_SALT), mask.selected.data(),
                                                mask.selected.size()) :
                                0;
    return true;
}

struct moe_measure_dataset_inventory {
    size_t records              = 0;
    size_t image_chat_records   = 0;
    size_t image_paired_records = 0;

    bool has_media() const { return image_chat_records != 0 || image_paired_records != 0; }
};

bool inspect_dataset_media(const measure_options & options, moe_measure_dataset_inventory & inventory, std::string & error) {
    inventory = {};
    for (const measure_options::dataset & dataset : options.datasets) {
        if (dataset.kind == measure_options::dataset_kind::text) {
            continue;
        }
        std::ifstream input(dataset.path);
        if (!input) {
            error = "failed to open chat dataset " + dataset.path;
            return false;
        }
        std::string line;
        size_t      line_number = 0;
        while (std::getline(input, line)) {
            line_number++;
            if (line.empty()) {
                continue;
            }
            try {
                const json record = json::parse(line);
                bool       has_images = false;
                if (dataset.kind == measure_options::dataset_kind::chat) {
                    if (!record.is_object() || !record.contains("messages")) {
                        error = "chat record must contain messages";
                        throw std::runtime_error(error);
                    }
                    if (!moe_measure_media_messages_have_images(record["messages"], has_images, error)) {
                        throw std::runtime_error(error);
                    }
                    inventory.image_chat_records += has_images;
                } else {
                    if (!record.is_object() || !record.contains("input")) {
                        error = "input JSONL record must be an object containing input";
                        throw std::runtime_error(error);
                    }
                    const json   empty_prefix = "";
                    const json & prefix       = record.contains("context_prefix") ? record["context_prefix"] :
                                                                                     empty_prefix;
                    const json & record_input = record["input"];
                    if ((!prefix.is_string() && !prefix.is_array()) ||
                        (!record_input.is_string() && !record_input.is_array())) {
                        error = "input and context_prefix must each be a string or message array";
                        throw std::runtime_error(error);
                    }
                    bool prefix_images = false;
                    bool input_images  = false;
                    if (prefix.is_array() &&
                        !moe_measure_media_messages_have_images(prefix, prefix_images, error)) {
                        throw std::runtime_error(error);
                    }
                    if (record_input.is_array() &&
                        !moe_measure_media_messages_have_images(record_input, input_images, error)) {
                        throw std::runtime_error(error);
                    }
                    has_images = prefix_images || input_images;
                    inventory.image_paired_records += has_images;
                }
                inventory.records++;
            } catch (const std::exception & exception) {
                error = dataset.path + ":" + std::to_string(line_number) + ": invalid JSON: " + exception.what();
                return false;
            }
        }
        if (!input.eof()) {
            error = "failed to read chat dataset " + dataset.path;
            return false;
        }
    }
    return true;
}

bool prepare_chat_prompt(const json &                  record,
                         const json &                  messages,
                         const common_chat_templates * templates,
                         const common_params &         params,
                         std::string &                 prompt,
                         std::string &                 error,
                         int                           add_generation_prompt = -1) {
    if (!record.is_object() || !messages.is_array()) {
        error = "chat messages must be an array";
        return false;
    }
    try {
        common_chat_templates_inputs inputs;
        inputs.messages  = common_chat_msgs_parse_oaicompat(messages);
        inputs.use_jinja = params.use_jinja;
        inputs.add_generation_prompt =
            add_generation_prompt >= 0 ? add_generation_prompt != 0 : record.value("add_generation_prompt", false);
        const bool default_thinking =
            params.enable_reasoning != 0 && common_chat_templates_support_enable_thinking(templates);
        inputs.enable_thinking = record.value("enable_thinking", default_thinking);
        if (record.contains("tools")) {
            inputs.tools = common_chat_tools_parse_oaicompat(record["tools"]);
        }
        inputs.chat_template_kwargs = params.default_template_kwargs;
        if (record.contains("chat_template_kwargs")) {
            if (!record["chat_template_kwargs"].is_object()) {
                error = "chat_template_kwargs must be an object";
                return false;
            }
            for (const auto & item : record["chat_template_kwargs"].items()) {
                inputs.chat_template_kwargs[item.key()] = item.value().dump();
            }
        }
        prompt = common_chat_templates_apply(templates, inputs).prompt;
    } catch (const std::exception & exception) {
        error = std::string("failed to parse or template chat record: ") + exception.what();
        return false;
    }
    return true;
}

struct moe_measure_paired_prompt {
    std::string                  prefix;
    std::string                  full;
    std::vector<moe_measure_media_blob> prefix_images;
    std::vector<moe_measure_media_blob> images;
};

bool prepare_paired_prompt(const json &                  record,
                           const measure_options &       options,
                           const common_chat_templates * templates,
                           const common_params &         params,
                           moe_measure_paired_prompt &          result,
                           std::string &                 error) {
    if (!record.is_object() || !record.contains("input")) {
        error = "input JSONL record must be an object containing input";
        return false;
    }
    const json   empty_prefix = "";
    const json & prefix       = record.contains("context_prefix") ? record["context_prefix"] : empty_prefix;
    const json & input        = record["input"];
    if ((!prefix.is_string() && !prefix.is_array()) || (!input.is_string() && !input.is_array())) {
        error = "input and context_prefix must each be a string or message array";
        return false;
    }

    json prefix_messages = prefix.is_array() ? prefix : json::array();
    json input_messages  = input.is_array() ? input : json::array();
    if (prefix.is_array() && !moe_measure_media_extract_images(prefix_messages, mtmd_default_marker(), options.media_path,
                                                        options.media_max_bytes, result.prefix_images, error)) {
        return false;
    }
    std::vector<moe_measure_media_blob> input_images;
    if (input.is_array() && !moe_measure_media_extract_images(input_messages, mtmd_default_marker(), options.media_path,
                                                       options.media_max_bytes, input_images, error)) {
        return false;
    }
    result.images = result.prefix_images;
    result.images.insert(result.images.end(), input_images.begin(), input_images.end());

    if (prefix.is_string() && input.is_string()) {
        result.prefix = prefix.get<std::string>();
        result.full   = result.prefix + input.get<std::string>();
    } else if (prefix.is_array() && input.is_string()) {
        if (!prepare_chat_prompt(record, prefix_messages, templates, params, result.prefix, error, 1)) {
            return false;
        }
        result.full = result.prefix + input.get<std::string>();
    } else if (prefix.is_array() && input.is_array()) {
        if (!prepare_chat_prompt(record, prefix_messages, templates, params, result.prefix, error, 0)) {
            return false;
        }
        json combined = prefix_messages;
        combined.insert(combined.end(), input_messages.begin(), input_messages.end());
        if (!prepare_chat_prompt(record, combined, templates, params, result.full, error)) {
            return false;
        }
    } else {
        result.prefix = prefix.get<std::string>();
        std::string rendered_input;
        if (!prepare_chat_prompt(record, input_messages, templates, params, rendered_input, error)) {
            return false;
        }
        result.full = result.prefix + rendered_input;
    }
    return true;
}

std::unique_ptr<moe_measure_measure_task> prepare_text_measurement(const moe_measure_measurement_header & header,
                                                            const llama_vocab *             vocab,
                                                            const moe_measure_vocab_mask &         vocab_mask,
                                                            uint64_t                        source_id,
                                                            std::vector<llama_token>        tokens,
                                                            std::unordered_set<uint64_t> &  reserved_hashes,
                                                            uint64_t &                      skipped) {
    if (tokens.empty()) {
        return nullptr;
    }
    if (llama_vocab_get_add_bos(vocab)) {
        tokens[0] = llama_vocab_bos(vocab);
    }
    std::vector<int32_t> tokens_i32(tokens.begin(), tokens.end());
    uint64_t             context_hash = moe_measure_hash_bytes(0, &header.model_signature, sizeof(header.model_signature));
    context_hash = moe_measure_hash_bytes(context_hash, &header.template_hash, sizeof(header.template_hash));
    context_hash = moe_measure_hash_bytes(context_hash, &header.output_role, sizeof(header.output_role));
    context_hash = moe_measure_hash_bytes(context_hash, &header.vocab_mask_hash, sizeof(header.vocab_mask_hash));
    context_hash = moe_measure_hash_bytes(context_hash, tokens_i32.data(), tokens_i32.size() * sizeof(int32_t));
    if (!reserved_hashes.insert(context_hash).second) {
        LOG_INF("moe-measure: skipping committed or queued text context %016llx (%zu tokens)\n",
                (unsigned long long) context_hash, tokens.size());
        skipped++;
        return nullptr;
    }

    auto task    = std::make_unique<moe_measure_measure_task>();
    task->tokens = std::move(tokens);
    task->token_mask.resize(task->tokens.size());
    size_t selected_tokens = 0;
    for (size_t i = 0; i < task->tokens.size(); ++i) {
        task->token_mask[i] = vocab_mask.includes(task->tokens[i]);
        selected_tokens += task->token_mask[i] != 0;
    }
    if (selected_tokens == 0) {
        LOG_INF("moe-measure: skipping text context %016llx because its tokens are excluded by the vocabulary mask\n",
                (unsigned long long) context_hash);
        return nullptr;
    }
    task->segments.push_back({ moe_measure_segment_kind::text, moe_measure_measure_destination::primary, 0, 0, task->tokens.size() });
    moe_measure_init_measurement_block(*task, header, source_id, context_hash, selected_tokens);
    uint64_t rolling = moe_measure_hash_bytes(0, &header.model_signature, sizeof(header.model_signature));
    size_t   output  = 0;
    for (size_t i = 0; i < task->tokens.size(); ++i) {
        const int32_t token = task->tokens[i];
        rolling             = moe_measure_hash_bytes(rolling, &token, sizeof(token));
        if (task->token_mask[i]) {
            task->block.token_hashes[output] = rolling;
            task->block.token_ids[output]    = token;
            output++;
        }
    }
    return task;
}

std::unique_ptr<moe_measure_measure_task> prepare_paired_text_measurement(const moe_measure_measurement_header & primary_header,
                                                                   const moe_measure_measurement_header * prefix_header,
                                                                   const llama_vocab *             vocab,
                                                                   const moe_measure_vocab_mask &         vocab_mask,
                                                                   uint64_t                        source_id,
                                                                   const std::string &             prefix_prompt,
                                                                   const std::string &             prompt,
                                                                   bool                            parse_special,
                                                                   size_t                          context_size,
                                                                   std::unordered_set<uint64_t> &  primary_hashes,
                                                                   std::unordered_set<uint64_t> *  prefix_hashes,
                                                                   uint64_t &                      skipped,
                                                                   std::string &                   error) {
    std::vector<llama_token> tokens = common_tokenize(vocab, prompt, true, parse_special);
    std::vector<llama_token> prefix_tokens;
    if (!prefix_prompt.empty()) {
        prefix_tokens = common_tokenize(vocab, prefix_prompt, true, parse_special);
    }
    size_t boundary = 0;
    while (boundary < tokens.size() && boundary < prefix_tokens.size() && tokens[boundary] == prefix_tokens[boundary]) {
        boundary++;
    }
    const size_t input_tokens = tokens.size() - boundary;
    if (input_tokens > context_size) {
        error = "complete input part exceeds the active per-slot model context";
        return nullptr;
    }

    size_t prefix_begin = 0;
    bool   preserve_bos = false;
    if (tokens.size() > context_size) {
        const size_t prefix_capacity = context_size - input_tokens;
        prefix_begin                 = boundary - std::min(boundary, prefix_capacity);
        preserve_bos =
            prefix_begin > 0 && prefix_capacity > 0 && !tokens.empty() && tokens[0] == llama_vocab_bos(vocab);
        if (preserve_bos) {
            prefix_begin = boundary - std::min(boundary - 1, prefix_capacity - 1);
        }
    }

    auto task    = std::make_unique<moe_measure_measure_task>();
    task->tokens = std::move(tokens);
    task->token_mask.resize(task->tokens.size());
    for (size_t i = 0; i < task->tokens.size(); ++i) {
        task->token_mask[i] = vocab_mask.includes(task->tokens[i]);
    }
    if (preserve_bos) {
        task->segments.push_back({ moe_measure_segment_kind::text, moe_measure_measure_destination::prefix, 0, 0, 1 });
    }
    if (prefix_begin < boundary) {
        task->segments.push_back(
            { moe_measure_segment_kind::text, moe_measure_measure_destination::prefix, 0, prefix_begin, boundary - prefix_begin });
    }
    if (boundary < task->tokens.size()) {
        task->segments.push_back({ moe_measure_segment_kind::text, moe_measure_measure_destination::primary, 0, boundary,
                                   task->tokens.size() - boundary });
    }
    if (task->segments.empty()) {
        error = "input record produces no decoder tokens";
        return nullptr;
    }

    uint64_t record_hash = moe_measure_hash_bytes(0, &primary_header.model_signature, sizeof(primary_header.model_signature));
    record_hash = moe_measure_hash_bytes(record_hash, &primary_header.template_hash, sizeof(primary_header.template_hash));
    record_hash =
        moe_measure_hash_bytes(record_hash, &primary_header.tokenization_hash, sizeof(primary_header.tokenization_hash));
    record_hash = moe_measure_hash_bytes(record_hash, &primary_header.vocab_mask_hash, sizeof(primary_header.vocab_mask_hash));
    for (const moe_measure_measure_segment & segment : task->segments) {
        const uint32_t destination = static_cast<uint32_t>(segment.destination);
        record_hash                = moe_measure_hash_bytes(record_hash, &destination, sizeof(destination));
        record_hash =
            moe_measure_hash_bytes(record_hash, task->tokens.data() + segment.begin, segment.count * sizeof(task->tokens[0]));
    }
    const uint32_t primary_role = MOE_MEASURE_OUTPUT_ROLE_PRIMARY;
    const uint32_t prefix_role  = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
    const uint64_t primary_hash = moe_measure_hash_bytes(record_hash, &primary_role, sizeof(primary_role));
    const uint64_t prefix_hash  = moe_measure_hash_bytes(record_hash, &prefix_role, sizeof(prefix_role));
    task->context_hash          = record_hash;

    size_t primary_count = 0;
    size_t prefix_count  = 0;
    for (const moe_measure_measure_segment & segment : task->segments) {
        for (size_t i = 0; i < segment.count; ++i) {
            if (!task->token_mask[segment.begin + i]) {
                continue;
            }
            if (segment.destination == moe_measure_measure_destination::primary) {
                primary_count++;
            } else if (segment.destination == moe_measure_measure_destination::prefix) {
                prefix_count++;
            }
        }
    }
    if (primary_count == 0) {
        error = "input part has no tokens selected by the vocabulary mask";
        return nullptr;
    }
    task->write_primary = primary_hashes.insert(primary_hash).second;
    task->write_prefix  = prefix_header != nullptr && prefix_count > 0 && prefix_hashes != nullptr &&
                          prefix_hashes->insert(prefix_hash).second;
    if (!task->write_primary && !task->write_prefix) {
        skipped++;
        return nullptr;
    }
    if (task->write_primary) {
        moe_measure_init_measurement_block(task->block, primary_header, source_id, primary_hash, primary_count, false);
    }
    if (task->write_prefix) {
        moe_measure_init_measurement_block(task->prefix_block, *prefix_header, source_id, prefix_hash, prefix_count, false);
    }

    uint64_t rolling = moe_measure_hash_bytes(0, &primary_header.model_signature, sizeof(primary_header.model_signature));
    size_t   primary_output = 0;
    size_t   prefix_output  = 0;
    for (const moe_measure_measure_segment & segment : task->segments) {
        for (size_t i = 0; i < segment.count; ++i) {
            const size_t  index = segment.begin + i;
            const int32_t token = task->tokens[index];
            rolling             = moe_measure_hash_bytes(rolling, &token, sizeof(token));
            if (!task->token_mask[index]) {
                continue;
            }
            if (segment.destination == moe_measure_measure_destination::primary && task->write_primary) {
                task->block.token_hashes[primary_output] = rolling;
                task->block.token_ids[primary_output]    = token;
                primary_output++;
            } else if (segment.destination == moe_measure_measure_destination::prefix && task->write_prefix) {
                task->prefix_block.token_hashes[prefix_output] = rolling;
                task->prefix_block.token_ids[prefix_output]    = token;
                prefix_output++;
            }
        }
    }
    return task;
}

bool prepare_mtmd_bitmaps(mtmd_context *                              mctx,
                          const std::vector<moe_measure_media_blob> &        images,
                          mtmd::bitmaps &                             bitmaps,
                          std::unordered_map<std::string, uint64_t> & image_digest_by_id,
                          std::string &                               error) {
    for (const moe_measure_media_blob & image : images) {
        mtmd_helper_bitmap_wrapper wrapper =
            mtmd_helper_bitmap_init_from_buf(mctx, image.bytes.data(), image.bytes.size(), false);
        if (wrapper.video_ctx != nullptr) {
            mtmd_helper_video_free(wrapper.video_ctx);
            if (wrapper.bitmap != nullptr) {
                mtmd_bitmap_free(wrapper.bitmap);
            }
            error = "video input is not supported";
            return false;
        }
        if (wrapper.bitmap == nullptr) {
            error = "failed to decode image bytes";
            return false;
        }
        if (mtmd_bitmap_is_audio(wrapper.bitmap)) {
            mtmd_bitmap_free(wrapper.bitmap);
            error = "audio input is not supported";
            return false;
        }
        std::ostringstream id;
        id << std::hex << image.digest;
        mtmd_bitmap_set_id(wrapper.bitmap, id.str().c_str());
        image_digest_by_id[id.str()] = image.digest;
        bitmaps.entries.emplace_back(wrapper.bitmap);
    }
    return true;
}

bool tokenize_mtmd_prompt(mtmd_context *      mctx,
                          const std::string & prompt,
                          bool                parse_special,
                          mtmd::bitmaps &     bitmaps,
                          mtmd_input_chunks * chunks,
                          std::string &       error) {
    mtmd_input_text                  input       = { prompt.c_str(), prompt.size(), true, parse_special };
    std::vector<const mtmd_bitmap *> bitmap_ptrs = bitmaps.c_ptr();
    const int32_t result = mtmd_tokenize(mctx, chunks, &input, bitmap_ptrs.data(), bitmap_ptrs.size());
    if (result != 0) {
        error = "mtmd failed to tokenize or preprocess the multimodal prompt (code " + std::to_string(result) + ")";
        return false;
    }
    return true;
}

std::unique_ptr<moe_measure_measure_task> prepare_multimodal_measurement(mtmd_context *                       mctx,
                                                                  const moe_measure_measurement_header &      header,
                                                                  const moe_measure_vocab_mask &              vocab_mask,
                                                                  uint64_t                             source_id,
                                                                  const std::string &                  prompt,
                                                                  const std::vector<moe_measure_media_blob> & images,
                                                                  size_t                               context_size,
                                                                  std::unordered_set<uint64_t> &       reserved_hashes,
                                                                  uint64_t &                           skipped,
                                                                  std::string &                        error) {
    const uint64_t context_hash = moe_measure_multimodal_context_hash(header, prompt, images);
    if (!reserved_hashes.insert(context_hash).second) {
        LOG_INF("moe-measure: skipping committed or queued multimodal context %016llx (%zu images)\n",
                (unsigned long long) context_hash, images.size());
        skipped++;
        return nullptr;
    }
    if (mctx == nullptr) {
        error = "image-bearing chat record requires --mmproj";
        return nullptr;
    }

    LOG_INF("moe-measure: preprocessing multimodal context %016llx (%zu images, %zu prompt bytes)\n",
            (unsigned long long) context_hash, images.size(), prompt.size());
    mtmd::bitmaps                             bitmaps;
    std::unordered_map<std::string, uint64_t> image_digest_by_id;
    if (!prepare_mtmd_bitmaps(mctx, images, bitmaps, image_digest_by_id, error)) {
        return nullptr;
    }

    auto task = std::make_unique<moe_measure_measure_task>(mtmd_input_chunks_init());
    if (!tokenize_mtmd_prompt(mctx, prompt, true, bitmaps, task->chunks.ptr.get(), error)) {
        return nullptr;
    }
    const llama_pos n_positions      = mtmd_helper_get_n_pos(task->chunks.ptr.get());
    const size_t    n_decoder_tokens = mtmd_helper_get_n_tokens(task->chunks.ptr.get());
    if (n_positions <= 0 || static_cast<size_t>(n_positions) > context_size || n_decoder_tokens > context_size) {
        error = "multimodal record exceeds the active per-slot model context";
        return nullptr;
    }

    const bool select_text     = moe_measure_scope_selects(header.input_scope, true, false);
    const bool select_media    = moe_measure_scope_selects(header.input_scope, true, true);
    size_t     selected_tokens = 0;
    task->chunk_token_masks.resize(task->chunks.size());
    for (size_t i = 0; i < task->chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = task->chunks[i];
        const auto               type  = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            error = "mtmd produced an unsupported audio chunk";
            return nullptr;
        }
        const size_t n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        const bool   selected = type == MTMD_INPUT_CHUNK_TYPE_TEXT ? select_text : select_media;
        task->segments.push_back(
            { type == MTMD_INPUT_CHUNK_TYPE_TEXT ? moe_measure_segment_kind::text : moe_measure_segment_kind::media,
              selected ? moe_measure_measure_destination::primary : moe_measure_measure_destination::none, i, 0, n_tokens });
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              count  = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
            task->chunk_token_masks[i].resize(count);
            for (size_t j = 0; j < count; ++j) {
                task->chunk_token_masks[i][j] = select_text && vocab_mask.includes(tokens[j]);
                selected_tokens += task->chunk_token_masks[i][j] != 0;
            }
        } else if (selected) {
            selected_tokens += n_tokens;
        }
    }
    if (selected_tokens == 0) {
        error = "multimodal scope selected no decoder tokens";
        return nullptr;
    }

    task->multimodal = true;
    moe_measure_init_measurement_block(*task, header, source_id, context_hash, selected_tokens);
    task->block.token_hashes.clear();
    task->block.token_hashes.reserve(selected_tokens);
    task->block.token_ids.clear();
    task->block.token_ids.reserve(selected_tokens);
    uint64_t  rolling    = moe_measure_hash_bytes(0, &header.model_signature, sizeof(header.model_signature));
    llama_pos prefix_pos = 0;
    for (size_t i = 0; i < task->chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = task->chunks[i];
        const auto               type  = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              n_tokens = 0;
            const llama_token * tokens   = mtmd_input_chunk_get_tokens_text(chunk, &n_tokens);
            for (size_t j = 0; j < n_tokens; ++j) {
                rolling = moe_measure_prefix_hash_text(rolling, tokens[j]);
                if (select_text && vocab_mask.includes(tokens[j])) {
                    task->block.token_hashes.push_back(rolling);
                    task->block.token_ids.push_back(tokens[j]);
                }
            }
        } else {
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            const size_t              n_tokens     = mtmd_input_chunk_get_n_tokens(chunk);
            const char *              id           = mtmd_input_chunk_get_id(chunk);
            const std::string         id_string    = id == nullptr ? std::string() : std::string(id);
            const auto                digest_it    = image_digest_by_id.find(id_string);
            const uint64_t            digest =
                digest_it == image_digest_by_id.end() ? moe_measure_hash_string(0, id_string) : digest_it->second;
            for (size_t j = 0; j < n_tokens; ++j) {
                const mtmd_decoder_pos pos = mtmd_image_tokens_get_decoder_pos(image_tokens, prefix_pos, j);
                rolling = moe_measure_prefix_hash_media(rolling, digest, static_cast<uint32_t>(i), static_cast<uint32_t>(j),
                                                 pos.t, pos.x, pos.y, pos.z);
                if (select_media) {
                    task->block.token_hashes.push_back(rolling);
                    task->block.token_ids.push_back(-1);
                }
            }
        }
        prefix_pos += mtmd_input_chunk_get_n_pos(chunk);
    }
    if (task->block.token_hashes.size() != selected_tokens) {
        error = "internal multimodal prefix hash count mismatch";
        return nullptr;
    }
    if (task->block.token_ids.size() != selected_tokens) {
        error = "internal multimodal token ID count mismatch";
        return nullptr;
    }
    LOG_INF("moe-measure: prepared multimodal context %016llx (%d positions, %zu selected tokens)\n",
            (unsigned long long) context_hash, n_positions, selected_tokens);
    return task;
}

struct moe_measure_mtmd_unit {
    moe_measure_segment_kind kind;
    size_t            chunk;
    size_t            token;
    uint64_t          identity;
};

std::vector<moe_measure_mtmd_unit> flatten_mtmd_chunks(const mtmd::input_chunks &                        chunks,
                                                const std::unordered_map<std::string, uint64_t> & image_digest_by_id) {
    std::vector<moe_measure_mtmd_unit> result;
    for (size_t i = 0; i < chunks.size(); ++i) {
        const mtmd_input_chunk * chunk    = chunks[i];
        const auto               type     = mtmd_input_chunk_get_type(chunk);
        const size_t             n_tokens = mtmd_input_chunk_get_n_tokens(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              count  = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
            for (size_t j = 0; j < count; ++j) {
                uint64_t identity = moe_measure_hash_string(0, "text");
                identity          = moe_measure_hash_bytes(identity, &tokens[j], sizeof(tokens[j]));
                result.push_back({ moe_measure_segment_kind::text, i, j, identity });
            }
        } else {
            const char *      id        = mtmd_input_chunk_get_id(chunk);
            const std::string id_string = id == nullptr ? std::string() : std::string(id);
            const auto        digest_it = image_digest_by_id.find(id_string);
            const uint64_t    digest =
                digest_it == image_digest_by_id.end() ? moe_measure_hash_string(0, id_string) : digest_it->second;
            for (size_t j = 0; j < n_tokens; ++j) {
                uint64_t       identity    = moe_measure_hash_string(0, "media");
                const uint64_t token_index = j;
                const uint64_t token_count = n_tokens;
                identity                   = moe_measure_hash_bytes(identity, &digest, sizeof(digest));
                identity                   = moe_measure_hash_bytes(identity, &token_index, sizeof(token_index));
                identity                   = moe_measure_hash_bytes(identity, &token_count, sizeof(token_count));
                result.push_back({ moe_measure_segment_kind::media, i, j, identity });
            }
        }
    }
    return result;
}

std::unique_ptr<moe_measure_measure_task> prepare_paired_multimodal_measurement(mtmd_context *                  mctx,
                                                                         const moe_measure_measurement_header & primary_header,
                                                                         const moe_measure_measurement_header * prefix_header,
                                                                         const llama_vocab *             vocab,
                                                                         const moe_measure_vocab_mask &         vocab_mask,
                                                                         uint64_t                        source_id,
                                                                         const moe_measure_paired_prompt &      prompt,
                                                                         bool                            parse_special,
                                                                         size_t                          context_size,
                                                                         std::unordered_set<uint64_t> &  primary_hashes,
                                                                         std::unordered_set<uint64_t> *  prefix_hashes,
                                                                         uint64_t &                      skipped,
                                                                         std::string &                   error) {
    if (mctx == nullptr) {
        error = "image-bearing input record requires --mmproj";
        return nullptr;
    }
    auto                                      task = std::make_unique<moe_measure_measure_task>(mtmd_input_chunks_init());
    mtmd::bitmaps                             full_bitmaps;
    std::unordered_map<std::string, uint64_t> full_digests;
    if (!prepare_mtmd_bitmaps(mctx, prompt.images, full_bitmaps, full_digests, error) ||
        !tokenize_mtmd_prompt(mctx, prompt.full, parse_special, full_bitmaps, task->chunks.ptr.get(), error)) {
        return nullptr;
    }

    std::vector<moe_measure_mtmd_unit> prefix_units;
    if (!prompt.prefix.empty()) {
        mtmd::bitmaps                             prefix_bitmaps;
        std::unordered_map<std::string, uint64_t> prefix_digests;
        mtmd::input_chunks                        prefix_chunks(mtmd_input_chunks_init());
        if (!prepare_mtmd_bitmaps(mctx, prompt.prefix_images, prefix_bitmaps, prefix_digests, error) ||
            !tokenize_mtmd_prompt(mctx, prompt.prefix, parse_special, prefix_bitmaps, prefix_chunks.ptr.get(), error)) {
            return nullptr;
        }
        prefix_units = flatten_mtmd_chunks(prefix_chunks, prefix_digests);
    }
    const std::vector<moe_measure_mtmd_unit> full_units = flatten_mtmd_chunks(task->chunks, full_digests);
    size_t                            boundary   = 0;
    while (boundary < full_units.size() && boundary < prefix_units.size() &&
           full_units[boundary].identity == prefix_units[boundary].identity) {
        boundary++;
    }
    if (boundary < full_units.size() && full_units[boundary].kind == moe_measure_segment_kind::media) {
        while (boundary > 0 && full_units[boundary - 1].chunk == full_units[boundary].chunk) {
            boundary--;
        }
    }

    std::vector<size_t> chunk_starts(task->chunks.size() + 1, 0);
    for (size_t i = 0; i < task->chunks.size(); ++i) {
        chunk_starts[i + 1] = chunk_starts[i] + mtmd_input_chunk_get_n_tokens(task->chunks[i]);
    }
    std::vector<moe_measure_measure_segment> prefix_segments;
    std::vector<moe_measure_measure_segment> primary_segments;
    size_t                            input_positions  = 0;
    size_t                            prefix_positions = 0;
    for (size_t i = 0; i < task->chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = task->chunks[i];
        const auto               type  = mtmd_input_chunk_get_type(chunk);
        if (type == MTMD_INPUT_CHUNK_TYPE_AUDIO) {
            error = "mtmd produced an unsupported audio chunk";
            return nullptr;
        }
        const moe_measure_segment_kind kind =
            type == MTMD_INPUT_CHUNK_TYPE_TEXT ? moe_measure_segment_kind::text : moe_measure_segment_kind::media;
        const size_t count          = mtmd_input_chunk_get_n_tokens(chunk);
        const size_t local_boundary = boundary <= chunk_starts[i]     ? 0 :
                                      boundary >= chunk_starts[i + 1] ? count :
                                                                        boundary - chunk_starts[i];
        if (local_boundary > 0) {
            prefix_segments.push_back({ kind, moe_measure_measure_destination::prefix, i, 0, local_boundary });
            prefix_positions += kind == moe_measure_segment_kind::text ? local_boundary : mtmd_input_chunk_get_n_pos(chunk);
        }
        if (local_boundary < count) {
            primary_segments.push_back(
                { kind, moe_measure_measure_destination::primary, i, local_boundary, count - local_boundary });
            input_positions +=
                kind == moe_measure_segment_kind::text ? count - local_boundary : mtmd_input_chunk_get_n_pos(chunk);
        }
    }
    const size_t input_decoder_tokens = full_units.size() - boundary;
    if (input_decoder_tokens > context_size || input_positions > context_size) {
        error = "complete input part exceeds the active per-slot model context";
        return nullptr;
    }

    const size_t prefix_token_capacity = context_size - input_decoder_tokens;
    const size_t prefix_pos_capacity   = context_size - input_positions;
    if (boundary <= prefix_token_capacity && prefix_positions <= prefix_pos_capacity) {
        task->segments = prefix_segments;
    } else {
        bool preserve_bos = false;
        if (!prefix_segments.empty() && prefix_segments.front().kind == moe_measure_segment_kind::text &&
            prefix_segments.front().begin == 0 && prefix_token_capacity > 0 && prefix_pos_capacity > 0) {
            size_t              count = 0;
            const llama_token * tokens =
                mtmd_input_chunk_get_tokens_text(task->chunks[prefix_segments.front().chunk], &count);
            preserve_bos = count > 0 && tokens[0] == llama_vocab_bos(vocab);
        }
        size_t                            remaining_tokens = prefix_token_capacity - (preserve_bos ? 1 : 0);
        size_t                            remaining_pos    = prefix_pos_capacity - (preserve_bos ? 1 : 0);
        std::vector<moe_measure_measure_segment> retained;
        for (auto it = prefix_segments.rbegin(); it != prefix_segments.rend(); ++it) {
            moe_measure_measure_segment segment = *it;
            if (preserve_bos && segment.kind == moe_measure_segment_kind::text && segment.chunk == 0 && segment.begin == 0) {
                segment.begin++;
                segment.count--;
            }
            if (segment.count == 0) {
                continue;
            }
            if (segment.kind == moe_measure_segment_kind::media) {
                const size_t n_pos = mtmd_input_chunk_get_n_pos(task->chunks[segment.chunk]);
                if (segment.count > remaining_tokens || n_pos > remaining_pos) {
                    break;
                }
                remaining_tokens -= segment.count;
                remaining_pos -= n_pos;
                retained.push_back(segment);
            } else {
                const size_t count = std::min({ segment.count, remaining_tokens, remaining_pos });
                if (count == 0) {
                    break;
                }
                segment.begin += segment.count - count;
                segment.count = count;
                retained.push_back(segment);
                remaining_tokens -= count;
                remaining_pos -= count;
                if (count < it->count) {
                    break;
                }
            }
        }
        if (preserve_bos) {
            task->segments.push_back({ moe_measure_segment_kind::text, moe_measure_measure_destination::prefix, 0, 0, 1 });
        }
        task->segments.insert(task->segments.end(), retained.rbegin(), retained.rend());
    }
    task->segments.insert(task->segments.end(), primary_segments.begin(), primary_segments.end());
    task->multimodal = true;
    task->chunk_token_masks.resize(task->chunks.size());

    size_t primary_count = 0;
    size_t prefix_count  = 0;
    for (size_t i = 0; i < task->chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = task->chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              count  = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
            task->chunk_token_masks[i].resize(count);
            for (size_t j = 0; j < count; ++j) {
                task->chunk_token_masks[i][j] = vocab_mask.includes(tokens[j]);
            }
        }
    }
    for (const moe_measure_measure_segment & segment : task->segments) {
        size_t selected = 0;
        if (segment.kind == moe_measure_segment_kind::media) {
            selected = segment.count;
        } else {
            for (size_t j = 0; j < segment.count; ++j) {
                selected += task->chunk_token_masks[segment.chunk][segment.begin + j] != 0;
            }
        }
        if (segment.destination == moe_measure_measure_destination::primary) {
            primary_count += selected;
        } else {
            prefix_count += selected;
        }
    }
    if (primary_count == 0) {
        error = "input part has no selected text or image positions";
        return nullptr;
    }

    uint64_t record_hash = moe_measure_hash_bytes(0, &primary_header.model_signature, sizeof(primary_header.model_signature));
    record_hash = moe_measure_hash_bytes(record_hash, &primary_header.template_hash, sizeof(primary_header.template_hash));
    record_hash =
        moe_measure_hash_bytes(record_hash, &primary_header.tokenization_hash, sizeof(primary_header.tokenization_hash));
    record_hash =
        moe_measure_hash_bytes(record_hash, &primary_header.projector_signature, sizeof(primary_header.projector_signature));
    record_hash =
        moe_measure_hash_bytes(record_hash, &primary_header.media_config_hash, sizeof(primary_header.media_config_hash));
    record_hash = moe_measure_hash_bytes(record_hash, &primary_header.media_pipeline_version,
                                  sizeof(primary_header.media_pipeline_version));
    record_hash = moe_measure_hash_bytes(record_hash, &primary_header.vocab_mask_hash, sizeof(primary_header.vocab_mask_hash));
    for (const moe_measure_measure_segment & segment : task->segments) {
        const uint32_t destination = static_cast<uint32_t>(segment.destination);
        record_hash                = moe_measure_hash_bytes(record_hash, &destination, sizeof(destination));
        const size_t unit          = chunk_starts[segment.chunk] + segment.begin;
        for (size_t i = 0; i < segment.count; ++i) {
            record_hash =
                moe_measure_hash_bytes(record_hash, &full_units[unit + i].identity, sizeof(full_units[unit + i].identity));
        }
    }
    const uint32_t primary_role = MOE_MEASURE_OUTPUT_ROLE_PRIMARY;
    const uint32_t prefix_role  = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
    const uint64_t primary_hash = moe_measure_hash_bytes(record_hash, &primary_role, sizeof(primary_role));
    const uint64_t prefix_hash  = moe_measure_hash_bytes(record_hash, &prefix_role, sizeof(prefix_role));
    task->context_hash          = record_hash;
    task->write_primary         = primary_hashes.insert(primary_hash).second;
    task->write_prefix          = prefix_header != nullptr && prefix_count > 0 && prefix_hashes != nullptr &&
                                  prefix_hashes->insert(prefix_hash).second;
    if (!task->write_primary && !task->write_prefix) {
        skipped++;
        return nullptr;
    }
    if (task->write_primary) {
        moe_measure_init_measurement_block(task->block, primary_header, source_id, primary_hash, primary_count, false);
    }
    if (task->write_prefix) {
        moe_measure_init_measurement_block(task->prefix_block, *prefix_header, source_id, prefix_hash, prefix_count, false);
    }

    uint64_t  rolling = moe_measure_hash_bytes(0, &primary_header.model_signature, sizeof(primary_header.model_signature));
    size_t    primary_output = 0;
    size_t    prefix_output  = 0;
    llama_pos prefix_pos     = 0;
    for (const moe_measure_measure_segment & segment : task->segments) {
        const mtmd_input_chunk * chunk = task->chunks[segment.chunk];
        if (segment.kind == moe_measure_segment_kind::text) {
            size_t              count  = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
            for (size_t j = 0; j < segment.count; ++j) {
                const size_t index = segment.begin + j;
                rolling            = moe_measure_prefix_hash_text(rolling, tokens[index]);
                if (!task->chunk_token_masks[segment.chunk][index]) {
                    continue;
                }
                if (segment.destination == moe_measure_measure_destination::primary && task->write_primary) {
                    task->block.token_hashes[primary_output] = rolling;
                    task->block.token_ids[primary_output]    = tokens[index];
                    primary_output++;
                } else if (segment.destination == moe_measure_measure_destination::prefix && task->write_prefix) {
                    task->prefix_block.token_hashes[prefix_output] = rolling;
                    task->prefix_block.token_ids[prefix_output]    = tokens[index];
                    prefix_output++;
                }
            }
            prefix_pos += segment.count;
        } else {
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            const char *              id           = mtmd_input_chunk_get_id(chunk);
            const std::string         id_string    = id == nullptr ? std::string() : std::string(id);
            const auto                digest_it    = full_digests.find(id_string);
            const uint64_t            digest =
                digest_it == full_digests.end() ? moe_measure_hash_string(0, id_string) : digest_it->second;
            for (size_t j = 0; j < segment.count; ++j) {
                const size_t           index = segment.begin + j;
                const mtmd_decoder_pos pos   = mtmd_image_tokens_get_decoder_pos(image_tokens, prefix_pos, index);
                rolling = moe_measure_prefix_hash_media(rolling, digest, static_cast<uint32_t>(segment.chunk),
                                                 static_cast<uint32_t>(index), pos.t, pos.x, pos.y, pos.z);
                if (segment.destination == moe_measure_measure_destination::primary && task->write_primary) {
                    task->block.token_hashes[primary_output] = rolling;
                    task->block.token_ids[primary_output]    = -1;
                    primary_output++;
                } else if (segment.destination == moe_measure_measure_destination::prefix && task->write_prefix) {
                    task->prefix_block.token_hashes[prefix_output] = rolling;
                    task->prefix_block.token_ids[prefix_output]    = -1;
                    prefix_output++;
                }
            }
            prefix_pos += mtmd_input_chunk_get_n_pos(chunk);
        }
    }
    return task;
}

bool task_soft_token_bytes(const moe_measure_measure_task & task, size_t n_embd, size_t & bytes) {
    bytes = 0;
    for (size_t i = 0; i < task.chunks.size(); ++i) {
        const mtmd_input_chunk * chunk = task.chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            continue;
        }
        size_t chunk_bytes = 0;
        if (!moe_measure_soft_token_bytes(mtmd_input_chunk_get_n_tokens(chunk), n_embd, chunk_bytes) ||
            bytes > std::numeric_limits<size_t>::max() - chunk_bytes) {
            return false;
        }
        bytes += chunk_bytes;
    }
    return true;
}

bool materialize_multimodal_task(moe_measure_measure_task & task,
                                 mtmd_context *      mctx,
                                 size_t              n_embd,
                                 moe_measure_collector &    collector,
                                 std::string &       error) {
    if (!task.multimodal || !task.chunks.ptr) {
        return true;
    }
    if (n_embd == 0) {
        error = "multimodal projector returned an invalid embedding width";
        return false;
    }
    const bool   mrope  = mtmd_decode_use_mrope(mctx);
    task.prepared_chunks.resize(task.chunks.size());
    for (size_t i = 0; i < task.chunks.size(); ++i) {
        const mtmd_input_chunk * chunk    = task.chunks[i];
        moe_measure_prepared_chunk &    prepared = task.prepared_chunks[i];
        if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            size_t              count  = 0;
            const llama_token * tokens = mtmd_input_chunk_get_tokens_text(chunk, &count);
            prepared.kind              = moe_measure_segment_kind::text;
            prepared.tokens.assign(tokens, tokens + count);
        }
    }

    size_t chunk_index = 0;
    while (chunk_index < task.chunks.size()) {
        if (mtmd_input_chunk_get_type(task.chunks[chunk_index]) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
            chunk_index++;
            continue;
        }
        mtmd::batch_ptr     batch(mtmd_batch_init(mctx));
        std::vector<size_t> encoded;
        size_t              next = chunk_index;
        for (; next < task.chunks.size(); ++next) {
            const mtmd_input_chunk * chunk = task.chunks[next];
            if (mtmd_input_chunk_get_type(chunk) == MTMD_INPUT_CHUNK_TYPE_TEXT) {
                break;
            }
            if (mtmd_batch_add_chunk(batch.get(), chunk) != 0) {
                break;
            }
            encoded.push_back(next);
        }
        if (encoded.empty()) {
            error = "mtmd projector batch rejected a media chunk";
            return false;
        }
        const auto projector_start = moe_measure_clock::now();
        if (mtmd_batch_encode(batch.get()) != 0) {
            error = "mtmd projector encoding failed";
            return false;
        }
        collector.perf.projector_ms += elapsed_ms(projector_start);
        for (size_t index : encoded) {
            const mtmd_input_chunk *  chunk        = task.chunks[index];
            const mtmd_image_tokens * image_tokens = mtmd_input_chunk_get_tokens_image(chunk);
            const size_t              n_tokens     = mtmd_input_chunk_get_n_tokens(chunk);
            float *                   source       = mtmd_batch_get_output_embd(batch.get(), chunk);
            if (source == nullptr || (mrope && image_tokens == nullptr)) {
                error = "mtmd projector returned invalid image embeddings";
                return false;
            }
            moe_measure_prepared_chunk & prepared = task.prepared_chunks[index];
            prepared.kind                  = moe_measure_segment_kind::media;
            prepared.n_embd                = n_embd;
            prepared.n_pos                 = mtmd_input_chunk_get_n_pos(chunk);
            prepared.non_causal            = mtmd_decode_use_non_causal(mctx, chunk);
            prepared.embeddings.assign(source, source + n_tokens * n_embd);
            if (mrope) {
                prepared.positions_zero.reserve(n_tokens);
                prepared.positions_one.reserve(n_tokens);
                for (size_t token = 0; token < n_tokens; ++token) {
                    prepared.positions_zero.push_back(mtmd_image_tokens_get_decoder_pos(image_tokens, 0, token));
                    prepared.positions_one.push_back(mtmd_image_tokens_get_decoder_pos(image_tokens, 1, token));
                }
            }
        }
        chunk_index = next;
    }
    task.chunks.ptr.reset();
    return true;
}

bool gib_to_bytes(double gib, size_t & bytes) {
    constexpr long double gib_bytes = 1024.0L * 1024.0L * 1024.0L;
    const long double     value     = static_cast<long double>(gib) * gib_bytes;
    if (!std::isfinite(gib) || gib <= 0.0 || value > std::numeric_limits<size_t>::max()) {
        return false;
    }
    bytes = static_cast<size_t>(value);
    return bytes > 0;
}

size_t default_soft_token_buffer_bytes() {
    ggml_backend_dev_t cpu   = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_CPU);
    size_t             free  = 0;
    size_t             total = 0;
    if (cpu != nullptr) {
        ggml_backend_dev_memory(cpu, &free, &total);
    }
    if (total == 0) {
        LOG_WRN("moe-measure: physical RAM detection failed; using a 1 GiB soft-token buffer\n");
        return size_t(1024) * 1024 * 1024;
    }
    return std::max<size_t>(1, total / 10);
}

class moe_measure_task_source {
  public:
    moe_measure_task_source(const measure_options &                      options,
                     const llama_vocab *                          vocab,
                     const common_chat_templates *                templates,
                     const common_params &                        params,
                     std::function<mtmd_context *(std::string &)> projector,
                     const moe_measure_measurement_header &              header,
                     const moe_measure_measurement_header *              prefix_header,
                     const moe_measure_vocab_mask &                      vocab_mask,
                     size_t                                       context_size,
                     std::vector<llama_token>                     text_tokens,
                     std::unordered_set<uint64_t> &               reserved_hashes,
                     std::unordered_set<uint64_t> *               prefix_reserved_hashes,
                     uint64_t &                                   skipped) :
        options(options),
        vocab(vocab),
        templates(templates),
        params(params),
        projector(std::move(projector)),
        header(header),
        prefix_header(prefix_header),
        vocab_mask(vocab_mask),
        context_size(context_size),
        text_tokens(std::move(text_tokens)),
        reserved_hashes(reserved_hashes),
        prefix_reserved_hashes(prefix_reserved_hashes),
        skipped(skipped) {}

    bool next(std::unique_ptr<moe_measure_measure_task> & task, bool & done, std::string & error) {
        task.reset();
        done = false;
        while (dataset_index < options.datasets.size() || pending_offset < pending_tokens.size()) {
            if (pending_offset < pending_tokens.size()) {
                const size_t             count = std::min(context_size, pending_tokens.size() - pending_offset);
                std::vector<llama_token> context(pending_tokens.begin() + pending_offset,
                                                 pending_tokens.begin() + pending_offset + count);
                pending_offset += count;
                task = prepare_text_measurement(header, vocab, vocab_mask, pending_source_id, std::move(context),
                                                reserved_hashes, skipped);
                if (pending_offset == pending_tokens.size()) {
                    pending_tokens.clear();
                    pending_offset = 0;
                }
                if (task) {
                    task->ordinal = next_ordinal++;
                    return true;
                }
                continue;
            }

            const measure_options::dataset & dataset = options.datasets[dataset_index];
            if (dataset.kind == measure_options::dataset_kind::text) {
                dataset_index++;
                if (text_started) {
                    continue;
                }
                text_started      = true;
                pending_tokens    = std::move(text_tokens);
                pending_offset    = 0;
                pending_source_id = text_source_id;
                continue;
            }

            if (!chat_input.is_open()) {
                chat_path = dataset.path;
                chat_input.open(chat_path);
                chat_line = 0;
                if (!chat_input) {
                    error = "failed to open chat dataset " + chat_path;
                    return false;
                }
                LOG_INF("moe-measure: reading chat dataset: %s\n", chat_path.c_str());
            }

            std::string line;
            if (!std::getline(chat_input, line)) {
                if (!chat_input.eof()) {
                    error = "failed to read chat dataset " + chat_path;
                    return false;
                }
                chat_input.close();
                dataset_index++;
                continue;
            }
            chat_line++;
            if (line.empty()) {
                continue;
            }
            try {
                LOG_INF("moe-measure: preparing JSONL record %s:%zu\n", chat_path.c_str(), chat_line);
                const json record = json::parse(line);
                if (dataset.kind == measure_options::dataset_kind::input_jsonl) {
                    moe_measure_paired_prompt paired;
                    if (!prepare_paired_prompt(record, options, templates, params, paired, error)) {
                        return contextualize(error);
                    }
                    uint64_t source_id = moe_measure_hash_string(moe_measure_hash_string(0, "input-jsonl"), paired.full);
                    for (const moe_measure_media_blob & image : paired.images) {
                        source_id = moe_measure_hash_bytes(source_id, &image.digest, sizeof(image.digest));
                    }
                    if (paired.images.empty()) {
                        task = prepare_paired_text_measurement(header, prefix_header, vocab, vocab_mask, source_id,
                                                               paired.prefix, paired.full, params.parse_special,
                                                               context_size, reserved_hashes, prefix_reserved_hashes,
                                                               skipped, error);
                    } else {
                        mtmd_context * mctx = projector(error);
                        if (mctx == nullptr) {
                            return contextualize(error);
                        }
                        task = prepare_paired_multimodal_measurement(
                            mctx, header, prefix_header, vocab, vocab_mask, source_id, paired, params.parse_special,
                            context_size, reserved_hashes, prefix_reserved_hashes, skipped, error);
                    }
                    if (!error.empty()) {
                        return contextualize(error);
                    }
                    if (task) {
                        task->ordinal = next_ordinal++;
                        return true;
                    }
                    continue;
                }
                if (!record.is_object() || !record.contains("messages")) {
                    error = "chat record must contain messages";
                    return contextualize(error);
                }
                json                         messages = record["messages"];
                std::vector<moe_measure_media_blob> images;
                if (!moe_measure_media_extract_images(messages, mtmd_default_marker(), options.media_path,
                                               options.media_max_bytes, images, error)) {
                    return contextualize(error);
                }
                std::string prompt;
                if (!prepare_chat_prompt(record, messages, templates, params, prompt, error)) {
                    return contextualize(error);
                }
                LOG_INF("moe-measure: rendered chat record %s:%zu to %zu bytes with %zu images\n", chat_path.c_str(),
                        chat_line, prompt.size(), images.size());
                uint64_t source_id = moe_measure_hash_string(moe_measure_hash_string(0, "chat"), prompt);
                for (const moe_measure_media_blob & image : images) {
                    source_id = moe_measure_hash_bytes(source_id, &image.digest, sizeof(image.digest));
                }
                if (images.empty()) {
                    pending_tokens    = common_tokenize(vocab, prompt, true, true);
                    pending_offset    = 0;
                    pending_source_id = source_id;
                    continue;
                }
                mtmd_context * mctx         = nullptr;
                const uint64_t context_hash = moe_measure_multimodal_context_hash(header, prompt, images);
                if (reserved_hashes.count(context_hash) == 0) {
                    mctx = projector(error);
                    if (mctx == nullptr) {
                        return contextualize(error);
                    }
                }
                task = prepare_multimodal_measurement(mctx, header, vocab_mask, source_id, prompt, images, context_size,
                                                      reserved_hashes, skipped, error);
                if (!error.empty()) {
                    return contextualize(error);
                }
                if (task) {
                    task->ordinal = next_ordinal++;
                    return true;
                }
            } catch (const std::exception & exception) {
                error = std::string("invalid JSON: ") + exception.what();
                return contextualize(error);
            }
        }
        done = true;
        return true;
    }

    void set_text_source_id(uint64_t value) { text_source_id = value; }

  private:
    bool contextualize(std::string & error) const {
        error = chat_path + ":" + std::to_string(chat_line) + ": " + error;
        return false;
    }

    const measure_options &                      options;
    const llama_vocab *                          vocab;
    const common_chat_templates *                templates;
    const common_params &                        params;
    std::function<mtmd_context *(std::string &)> projector;
    const moe_measure_measurement_header &              header;
    const moe_measure_measurement_header *              prefix_header;
    const moe_measure_vocab_mask &                      vocab_mask;
    size_t                                       context_size;
    std::vector<llama_token>                     text_tokens;
    std::unordered_set<uint64_t> &               reserved_hashes;
    std::unordered_set<uint64_t> *               prefix_reserved_hashes;
    uint64_t &                                   skipped;
    size_t                                       dataset_index  = 0;
    bool                                         text_started   = false;
    uint64_t                                     text_source_id = 0;
    std::vector<llama_token>                     pending_tokens;
    size_t                                       pending_offset    = 0;
    uint64_t                                     pending_source_id = 0;
    std::ifstream                                chat_input;
    std::string                                  chat_path;
    size_t                                       chat_line    = 0;
    uint64_t                                     next_ordinal = 0;
};

bool finish_collection(moe_measure_collector & collector, bool collecting, std::string & error) {
    if (!collecting) {
        return true;
    }
    if (!collector.finish()) {
        error = collector.error;
        return false;
    }
    return true;
}

int decode_text_slots(llama_context *                  ctx,
                      moe_measure_collector &                 collector,
                      std::vector<moe_measure_measure_slot> & slots,
                      size_t                           capacity,
                      std::string &                    error) {
    std::vector<moe_measure_measure_slot *> candidates = moe_measure_select_text_slots(slots);
    std::vector<size_t>              remaining;
    for (const moe_measure_measure_slot * slot : candidates) {
        remaining.push_back(moe_measure_slot_segment_tokens(*slot) - slot->offset);
    }
    if (candidates.empty()) {
        return 0;
    }
    const std::vector<size_t>        counts = moe_measure_allocate_slot_tokens(remaining, capacity);
    const size_t                     total  = std::accumulate(counts.begin(), counts.end(), size_t(0));
    if (total == 0) {
        error = "internal text decoder batch contains no tokens";
        return -1;
    }
    llama_batch                      batch  = llama_batch_init(total, 0, 1);
    std::vector<moe_measure_collect_target> targets;
    targets.reserve(total);
    std::vector<size_t> primary_added(candidates.size(), 0);
    std::vector<size_t> prefix_added(candidates.size(), 0);
    bool                collecting = false;
    for (size_t i = 0; i < candidates.size(); ++i) {
        moe_measure_measure_slot &          slot           = *candidates[i];
        const moe_measure_measure_segment & segment        = slot.task->segments[slot.segment];
        size_t                       n_chunk_tokens = 0;
        const llama_token *          tokens         = nullptr;
        if (slot.task->multimodal) {
            const moe_measure_prepared_chunk * chunk = moe_measure_slot_prepared_chunk(slot);
            if (chunk != nullptr) {
                tokens         = chunk->tokens.data();
                n_chunk_tokens = chunk->tokens.size();
            }
        } else {
            tokens         = slot.task->tokens.data();
            n_chunk_tokens = slot.task->tokens.size();
        }
        if (tokens == nullptr || segment.begin + slot.offset + counts[i] > n_chunk_tokens) {
            llama_batch_free(batch);
            error = "internal text slot range mismatch";
            return -1;
        }
        for (size_t j = 0; j < counts[i]; ++j) {
            common_batch_add(batch, tokens[segment.begin + slot.offset + j], slot.n_past + j, { slot.seq_id }, true);
            const bool token_selected = moe_measure_slot_text_token_selected(slot, slot.offset + j);
            if (token_selected && segment.destination == moe_measure_measure_destination::primary &&
                slot.task->write_primary) {
                targets.push_back({ &slot.task->block, slot.primary_output + primary_added[i]++ });
                collecting = true;
            } else if (token_selected && segment.destination == moe_measure_measure_destination::prefix &&
                       slot.task->write_prefix) {
                targets.push_back({ &slot.task->prefix_block, slot.prefix_output + prefix_added[i]++ });
                collecting = true;
            } else {
                targets.push_back({});
            }
        }
    }
    if (collecting) {
        collector.begin(std::move(targets));
    } else {
        collector.disable();
    }
    LOG_DBG("moe-measure: decoding %zu text tokens across %zu measurement slots\n", total,
            std::count_if(counts.begin(), counts.end(), [](size_t count) { return count > 0; }));
    const auto    decode_start  = moe_measure_clock::now();
    const int32_t decode_result = llama_decode(ctx, batch);
    collector.perf.decode_ms += elapsed_ms(decode_start);
    collector.perf.decoded_tokens += total;
    llama_batch_free(batch);
    if (decode_result != 0) {
        error = "model evaluation failed for a parallel text batch";
        return -1;
    }
    if (!finish_collection(collector, collecting, error)) {
        return -1;
    }
    for (size_t i = 0; i < candidates.size(); ++i) {
        moe_measure_measure_slot & slot = *candidates[i];
        slot.offset += counts[i];
        slot.n_past += counts[i];
        slot.primary_output += primary_added[i];
        slot.prefix_output += prefix_added[i];
        if (slot.offset == moe_measure_slot_segment_tokens(slot)) {
            slot.segment++;
            slot.offset = 0;
        }
    }
    return 1;
}

int decode_media_slots(llama_context *                  ctx,
                       moe_measure_collector &                 collector,
                       std::vector<moe_measure_measure_slot> & slots,
                       size_t                           capacity,
                       bool                             non_causal,
                       std::string &                    error) {
    std::vector<moe_measure_measure_slot *> candidates;
    for (moe_measure_measure_slot & slot : slots) {
        if (!slot.task || slot.segment >= slot.task->segments.size() ||
            slot.task->segments[slot.segment].kind != moe_measure_segment_kind::media) {
            continue;
        }
        const moe_measure_prepared_chunk * chunk = moe_measure_slot_prepared_chunk(slot);
        if (chunk == nullptr || chunk->kind != moe_measure_segment_kind::media) {
            error = "internal prepared media chunk is missing";
            return -1;
        }
        const bool slot_non_causal = chunk->non_causal;
        if (slot_non_causal == non_causal) {
            candidates.push_back(&slot);
            if (non_causal) {
                break;
            }
        }
    }
    if (candidates.empty()) {
        return 0;
    }

    std::vector<moe_measure_measure_slot *> encoded = std::move(candidates);
    const moe_measure_prepared_chunk *      first   = moe_measure_slot_prepared_chunk(*encoded.front());
    const size_t                     n_embd  = first->n_embd;
    const size_t                     n_pos   = first->positions_zero.empty() ? 1 : 4;
    while (!encoded.empty()) {
        std::vector<size_t> remaining;
        remaining.reserve(encoded.size());
        for (const moe_measure_measure_slot * slot : encoded) {
            remaining.push_back(moe_measure_slot_segment_tokens(*slot) - slot->offset);
        }
        const std::vector<size_t>        counts = moe_measure_allocate_slot_tokens(remaining, capacity);
        const size_t                     total  = std::accumulate(counts.begin(), counts.end(), size_t(0));
        if (total == 0) {
            error = "internal media decoder batch contains no tokens";
            return -1;
        }
        std::vector<float>               embeddings(total * n_embd);
        std::vector<llama_pos>           positions(total * n_pos);
        std::vector<int32_t>             n_seq_id(total, 1);
        std::vector<llama_seq_id>        seq_values(total);
        std::vector<llama_seq_id *>      seq_ids(total + 1, nullptr);
        std::vector<int8_t>              logits(total, true);
        std::vector<moe_measure_collect_target> targets;
        targets.reserve(total);
        std::vector<size_t>                     primary_added(encoded.size(), 0);
        std::vector<size_t>                     prefix_added(encoded.size(), 0);
        bool                                    collecting = false;
        size_t                                  row        = 0;
        std::unordered_set<moe_measure_measure_slot *> completed;
        for (size_t i = 0; i < encoded.size(); ++i) {
            moe_measure_measure_slot &          slot    = *encoded[i];
            const moe_measure_prepared_chunk *  chunk   = moe_measure_slot_prepared_chunk(slot);
            const moe_measure_measure_segment & segment = slot.task->segments[slot.segment];
            if (chunk == nullptr || n_embd == 0 || chunk->n_embd != n_embd ||
                (n_pos == 4 && (segment.begin + segment.count > chunk->positions_zero.size() ||
                                segment.begin + segment.count > chunk->positions_one.size()))) {
                error = "incompatible prepared media chunks in decoder batch";
                return -1;
            }
            for (size_t j = 0; j < counts[i]; ++j, ++row) {
                const size_t token = segment.begin + slot.offset + j;
                if (token >= chunk->embeddings.size() / n_embd) {
                    error = "internal prepared media embedding range mismatch";
                    return -1;
                }
                memcpy(embeddings.data() + row * n_embd, chunk->embeddings.data() + token * n_embd,
                       n_embd * sizeof(float));
                if (n_pos == 1) {
                    positions[row] = slot.n_past + token;
                } else {
                    const mtmd_decoder_pos & zero = chunk->positions_zero[token];
                    const mtmd_decoder_pos & one  = chunk->positions_one[token];
                    const mtmd_decoder_pos   pos  = {
                        moe_measure_resolve_position(zero.t, one.t, slot.n_past),
                        moe_measure_resolve_position(zero.x, one.x, slot.n_past),
                        moe_measure_resolve_position(zero.y, one.y, slot.n_past),
                        moe_measure_resolve_position(zero.z, one.z, slot.n_past),
                    };
                    positions[row]             = pos.t;
                    positions[total + row]     = pos.y;
                    positions[2 * total + row] = pos.x;
                    positions[3 * total + row] = pos.z;
                }
                seq_values[row] = slot.seq_id;
                seq_ids[row]    = &seq_values[row];
                if (segment.destination == moe_measure_measure_destination::primary && slot.task->write_primary) {
                    targets.push_back({ &slot.task->block, slot.primary_output + primary_added[i]++ });
                    collecting = true;
                } else if (segment.destination == moe_measure_measure_destination::prefix && slot.task->write_prefix) {
                    targets.push_back({ &slot.task->prefix_block, slot.prefix_output + prefix_added[i]++ });
                    collecting = true;
                } else {
                    targets.push_back({});
                }
            }
        }
        llama_batch batch = {
            static_cast<int32_t>(total),
            nullptr,
            embeddings.data(),
            positions.data(),
            n_seq_id.data(),
            seq_ids.data(),
            logits.data(),
        };
        if (collecting) {
            collector.begin(std::move(targets));
        } else {
            collector.disable();
        }
        LOG_DBG("moe-measure: decoding %zu media tokens across %zu measurement slots%s\n", total,
                std::count_if(counts.begin(), counts.end(), [](size_t count) { return count > 0; }),
                non_causal ? " (non-causal serialized)" : "");
        if (non_causal) {
            llama_set_causal_attn(ctx, false);
        }
        if (collector.mode == measure_options::collector_mode::cpu) {
            llama_set_embeddings(ctx, true);
        }
        const auto    decode_start  = moe_measure_clock::now();
        const int32_t decode_result = llama_decode(ctx, batch);
        collector.perf.decode_ms += elapsed_ms(decode_start);
        collector.perf.decoded_tokens += total;
        if (collector.mode == measure_options::collector_mode::cpu) {
            llama_set_embeddings(ctx, false);
        }
        if (non_causal) {
            llama_set_causal_attn(ctx, true);
        }
        if (decode_result != 0) {
            error = "model evaluation failed for a parallel media batch";
            return -1;
        }
        if (!finish_collection(collector, collecting, error)) {
            return -1;
        }
        for (size_t i = 0; i < encoded.size(); ++i) {
            moe_measure_measure_slot & slot = *encoded[i];
            slot.offset += counts[i];
            slot.primary_output += primary_added[i];
            slot.prefix_output += prefix_added[i];
            if (slot.offset == moe_measure_slot_segment_tokens(slot)) {
                const moe_measure_prepared_chunk * chunk = moe_measure_slot_prepared_chunk(slot);
                slot.n_past += chunk->n_pos;
                slot.segment++;
                slot.offset = 0;
                completed.insert(&slot);
            }
        }
        encoded.erase(std::remove_if(encoded.begin(), encoded.end(),
                                     [&](moe_measure_measure_slot * slot) { return completed.count(slot) != 0; }),
                      encoded.end());
    }
    return 1;
}

struct moe_measure_measure_output {
    std::string              path;
    moe_measure_measurement_header  header;
    moe_measure_measurement_summary summary;
    uint64_t                 measured = 0;
};

bool run_parallel_measurements(std::vector<std::unique_ptr<moe_measure_measure_task>> & tasks,
                               llama_context *                                   ctx,
                               moe_measure_collector &                                  collector,
                               moe_measure_measure_output &                             primary,
                               moe_measure_measure_output *                             prefix,
                               size_t                                            n_slots,
                               size_t                                            capacity,
                               std::string &                                     error) {
    std::vector<moe_measure_measure_slot> slots(n_slots);
    for (size_t i = 0; i < slots.size(); ++i) {
        slots[i].seq_id = static_cast<llama_seq_id>(i);
    }
    std::map<uint64_t, std::unique_ptr<moe_measure_measure_task>> pending;
    const uint64_t                                         first_ordinal = tasks.empty() ? 0 : tasks.front()->ordinal;
    uint64_t                                               next_commit   = first_ordinal;
    uint64_t                                               refilled      = 0;
    size_t                                                 task_index    = 0;
    size_t                                                 stage         = 0;
    moe_measure_clock::time_point                                 last_progress = moe_measure_clock::now();
    llama_memory_clear(llama_get_memory(ctx), true);

    auto flush = [&]() {
        while (true) {
            auto it = pending.find(next_commit);
            if (it == pending.end()) {
                return true;
            }
            moe_measure_measure_task & task         = *it->second;
            const auto          commit_start = moe_measure_clock::now();
            if (task.write_primary) {
                if (!moe_measure_measurement_append(primary.path, primary.header, task.block, error)) {
                    return false;
                }
                primary.summary.context_hashes.insert(task.block.context_hash);
                primary.measured++;
            }
            if (task.write_prefix) {
                if (prefix == nullptr ||
                    !moe_measure_measurement_append(prefix->path, prefix->header, task.prefix_block, error)) {
                    return false;
                }
                prefix->summary.context_hashes.insert(task.prefix_block.context_hash);
                prefix->measured++;
            }
            collector.perf.commit_ms += elapsed_ms(commit_start);
            LOG_DBG("moe-measure: committed input order %llu (primary %s, prefix %s)\n", (unsigned long long) task.ordinal,
                    task.write_primary ? "appended" : "present", task.write_prefix ? "appended" : "absent or present");
            pending.erase(it);
            next_commit++;
            if (next_commit % 100 == 0) {
                LOG_INF("moe-measure: committed %llu input records, %zu completed records buffered\n",
                        (unsigned long long) next_commit, pending.size());
            }
        }
    };

    auto retire = [&]() {
        for (moe_measure_measure_slot & slot : slots) {
            if (!slot.task || slot.segment != slot.task->segments.size()) {
                continue;
            }
            if ((slot.task->write_primary && slot.primary_output != slot.task->block.n_tokens) ||
                (slot.task->write_prefix && slot.prefix_output != slot.task->prefix_block.n_tokens)) {
                error = "internal selected-token count mismatch in completed measurement slot";
                return false;
            }
            llama_memory_seq_rm(llama_get_memory(ctx), slot.seq_id, -1, -1);
            pending.emplace(slot.task->ordinal, std::move(slot.task));
            slot.segment        = 0;
            slot.offset         = 0;
            slot.n_past         = 0;
            slot.primary_output = 0;
            slot.prefix_output  = 0;
        }
        return flush();
    };

    auto fill = [&]() {
        if (task_index == tasks.size() || pending.size() >= n_slots) {
            return true;
        }
        const std::vector<moe_measure_measure_slot *> empty =
            moe_measure_select_empty_slots(slots, tasks.size() - task_index);
        for (moe_measure_measure_slot * selected : empty) {
            moe_measure_measure_slot & slot = *selected;
            std::unique_ptr<moe_measure_measure_task> task = std::move(tasks[task_index++]);
            if (task->write_primary) {
                moe_measure_prepare_measurement_observations(task->block, primary.header);
            }
            if (task->write_prefix) {
                if (prefix == nullptr) {
                    error = "internal prefix output is missing";
                    return false;
                }
                moe_measure_prepare_measurement_observations(task->prefix_block, prefix->header);
            }
            llama_memory_seq_rm(llama_get_memory(ctx), slot.seq_id, -1, -1);
            LOG_DBG("moe-measure: assigned context %016llx to measurement slot %d\n", (unsigned long long) task->context_hash,
                    slot.seq_id);
            slot.task = std::move(task);
            refilled++;
        }
        return true;
    };

    while (true) {
        if (!retire() || !fill()) {
            return false;
        }
        size_t active = 0;
        for (const moe_measure_measure_slot & slot : slots) {
            active += slot.task != nullptr;
        }
        if (active == 0) {
            if (task_index != tasks.size()) {
                continue;
            }
            if (!flush()) {
                return false;
            }
            if (!pending.empty()) {
                error = "internal ordered measurement commit gap";
                return false;
            }
            return true;
        }

        bool decoded = false;
        for (size_t attempt = 0; attempt < 3 && !decoded; ++attempt) {
            const size_t current = (stage + attempt) % 3;
            int          result  = 0;
            if (current == 0) {
                result = decode_text_slots(ctx, collector, slots, capacity, error);
            } else {
                result = decode_media_slots(ctx, collector, slots, capacity, current == 2, error);
            }
            if (result < 0) {
                return false;
            }
            if (result > 0) {
                decoded = true;
                stage   = (current + 1) % 3;
            }
        }
        if (!decoded) {
            error = "parallel measurement scheduler found no runnable slot";
            return false;
        }
        if (elapsed_ms(last_progress) >= 5000.0) {
            LOG_INF("moe-measure: progress: %llu committed, %llu slots filled, %zu/%zu active, %zu completed buffered\n",
                    (unsigned long long) next_commit, (unsigned long long) refilled, active, n_slots, pending.size());
            last_progress = moe_measure_clock::now();
        }
    }
}

void log_vocab_mask_mismatch_sources(const measure_options & options, uint64_t requested_hash) {
    if (requested_hash == 0) {
        LOG_ERR("error:   requested run has no vocabulary mask; the existing measurement does\n");
        return;
    }
    if (options.exclude_special_tokens) {
        LOG_ERR("error:   requested vocabulary mask includes --exclude-special-tokens\n");
    }
    for (const std::string & path : options.measured_vocab_files) {
        LOG_ERR("error:   requested vocabulary mask includes --measured-vocab %s\n", path.c_str());
    }
    for (const std::string & path : options.excluded_vocab_files) {
        LOG_ERR("error:   requested vocabulary mask includes --excluded-vocab %s\n", path.c_str());
    }
}

bool reconcile_existing_measurement(const std::string &        path,
                                    const char *               description,
                                    bool                       has_media,
                                    const measure_options &    options,
                                    moe_measure_measurement_header &  requested,
                                    moe_measure_measurement_summary & summary,
                                    std::string &              error) {
    if (!moe_measure_measurement_read(path, summary, {}, error)) {
        return false;
    }
    requested.measurement_id = summary.header.measurement_id;
    if (summary.n_blocks == 0) {
        const std::vector<moe_measure_measurement_header_mismatch> mismatches =
            moe_measure_measurement_header_mismatches(requested, summary.header);
        if (!mismatches.empty()) {
            LOG_INF("moe-measure: replacing empty %s header with the requested configuration: %s\n", description,
                    path.c_str());
            if (!moe_measure_measurement_create(path, requested, error) ||
                !moe_measure_measurement_read(path, summary, {}, error)) {
                return false;
            }
        }
        return true;
    }
    if (!has_media && summary.has_media_tokens) {
        error = "existing " + std::string(description) + " " + path +
                " contains soft-token observations but the requested datasets contain no media";
        return false;
    }
    if (!has_media && summary.header.projector_signature != 0) {
        error = "existing " + std::string(description) + " " + path +
                " records a multimodal projector but the requested datasets contain no media";
        return false;
    }
    const std::vector<moe_measure_measurement_header_mismatch> mismatches =
        moe_measure_measurement_header_mismatches(requested, summary.header, has_media);
    if (!mismatches.empty()) {
        LOG_ERR("error: existing %s %s is incompatible with this model or measurement configuration:\n",
                description, path.c_str());
        bool vocab_mismatch = false;
        for (const moe_measure_measurement_header_mismatch & mismatch : mismatches) {
            LOG_ERR("error:   %s: existing %s, requested %s\n", mismatch.field.c_str(),
                    mismatch.existing.c_str(), mismatch.requested.c_str());
            vocab_mismatch |= mismatch.field == "vocab_mask_hash";
        }
        if (vocab_mismatch) {
            log_vocab_mask_mismatch_sources(options, requested.vocab_mask_hash);
        }
        error.clear();
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char ** argv) {
    std::setlocale(LC_NUMERIC, "C");
    common_init();

    measure_options     options;
    std::vector<char *> common_argv;
    std::string         error;
    if (!parse_custom_args(argc, argv, options, common_argv, error)) {
        LOG_ERR("error: %s\n", error.c_str());
        print_usage(argv[0]);
        return 1;
    }

    common_params params;
    params.n_ctx    = 512;
    params.escape   = false;
    params.warmup   = false;
    moe_measure_collector collector;
    params.cb_eval                     = collector_callback;
    params.cb_eval_user_data           = &collector;
    params.cb_eval_row_order           = collector_row_order_callback;
    params.cb_eval_row_order_user_data = &collector;
    if (!common_params_parse(static_cast<int>(common_argv.size()), common_argv.data(), params, LLAMA_EXAMPLE_IMATRIX)) {
        return 1;
    }
    const int template_overrides =
        !options.chat_template.empty() + !options.chat_template_file.empty() + !options.jinja_template_file.empty();
    if (template_overrides > 1) {
        LOG_ERR("error: --chat-template, --chat-template-file, and --jinja-template are mutually exclusive\n");
        return 1;
    }
    if (!options.jinja_template_file.empty() && options.use_jinja == 0) {
        LOG_ERR("error: --jinja-template cannot be combined with --no-jinja\n");
        return 1;
    }
    const std::string template_file =
        !options.jinja_template_file.empty() ? options.jinja_template_file : options.chat_template_file;
    if (!template_file.empty()) {
        if (!read_file(template_file, params.chat_template, error, "Jinja template")) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
        if (params.chat_template.empty()) {
            LOG_ERR("error: Jinja template %s is empty\n", template_file.c_str());
            return 1;
        }
    }
    if (!options.chat_template.empty()) {
        params.chat_template = options.chat_template;
    }
    if (options.use_jinja >= 0) {
        params.use_jinja = options.use_jinja != 0;
    }
    if (!options.jinja_template_file.empty()) {
        params.use_jinja = true;
    }
    if (options.enable_reasoning >= -1) {
        params.enable_reasoning = options.enable_reasoning;
        if (options.enable_reasoning >= 0) {
            params.default_template_kwargs["enable_thinking"] = options.enable_reasoning ? "true" : "false";
        }
    }
    if (!options.chat_template_kwargs.empty()) {
        try {
            const json kwargs = json::parse(options.chat_template_kwargs);
            if (!kwargs.is_object()) {
                throw std::runtime_error("value is not a JSON object");
            }
            for (const auto & item : kwargs.items()) {
                params.default_template_kwargs[item.key()] = item.value().dump();
            }
        } catch (const std::exception & exception) {
            LOG_ERR("error: invalid --chat-template-kwargs: %s\n", exception.what());
            return 1;
        }
    }
    if (params.model.path.empty() || params.out_file.empty() || params.n_ctx <= 0 || params.n_parallel <= 0) {
        LOG_ERR("error: model, output, context size, and parallel slot count are required\n");
        return 1;
    }
    params.expert_output_capture      = true;
    params.expert_output_capture_only = true;
    if (!options.prefix_output.empty()) {
        std::error_code             prefix_path_error;
        std::error_code             primary_path_error;
        const std::filesystem::path resolved_prefix =
            std::filesystem::weakly_canonical(options.prefix_output, prefix_path_error);
        const std::filesystem::path resolved_primary =
            std::filesystem::weakly_canonical(params.out_file, primary_path_error);
        if (prefix_path_error || primary_path_error) {
            LOG_ERR("error: failed to resolve measurement output paths\n");
            return 1;
        }
        std::error_code equivalent_error;
        const bool      same_existing_file =
            std::filesystem::exists(resolved_prefix) && std::filesystem::exists(resolved_primary) &&
            std::filesystem::equivalent(resolved_prefix, resolved_primary, equivalent_error);
        if (equivalent_error) {
            LOG_ERR("error: failed to compare measurement output paths\n");
            return 1;
        }
        if (resolved_prefix == resolved_primary || same_existing_file) {
            LOG_ERR("error: --prefix-output must not resolve to the primary output path\n");
            return 1;
        }
    }
    const int32_t requested_context_size = params.n_ctx;
    int32_t       aggregate_context_size = 0;
    if (!moe_measure_expand_context_size(requested_context_size, params.n_parallel, aggregate_context_size)) {
        LOG_ERR("error: per-slot context size multiplied by --parallel is too large\n");
        return 1;
    }
    params.n_ctx = aggregate_context_size;

    const char * template_description = "embedded model template";
    if (!options.jinja_template_file.empty()) {
        template_description = options.jinja_template_file.c_str();
    } else if (!options.chat_template_file.empty()) {
        template_description = options.chat_template_file.c_str();
    } else if (!options.chat_template.empty()) {
        template_description = "inline template override";
    }
    LOG_INF("moe-measure: model: %s\n", params.model.path.c_str());
    LOG_INF("moe-measure: output: %s\n", params.out_file.c_str());
    if (!options.prefix_output.empty()) {
        LOG_INF("moe-measure: prefix output: %s\n", options.prefix_output.c_str());
    }
    LOG_INF("moe-measure: datasets: %zu text files, %zu chat files, %zu paired input files\n", options.text_files.size(),
            options.chat_files.size(), options.input_jsonl_files.size());
    LOG_INF("moe-measure: chat template: %s (%s engine)\n", template_description, params.use_jinja ? "Jinja" : "legacy");
    LOG_INF("moe-measure: slots: %d with %d requested context tokens each\n", params.n_parallel, requested_context_size);
    LOG_INF("moe-measure: collector mode: %s\n",
            options.collector == measure_options::collector_mode::device ? "device" : "cpu");
    LOG_INF("moe-measure: inspecting JSONL datasets for media inputs\n");
    moe_measure_dataset_inventory inventory;
    if (!inspect_dataset_media(options, inventory, error)) {
        LOG_ERR("error: %s\n", error.c_str());
        return 1;
    }
    LOG_INF("moe-measure: dataset inventory: %zu JSONL records, %zu image chat records, %zu image paired records\n",
            inventory.records, inventory.image_chat_records, inventory.image_paired_records);
    const bool has_media = inventory.has_media();
    if (has_media && options.mmproj.empty()) {
        LOG_ERR("error: image-bearing datasets require --mmproj\n");
        return 1;
    }
    const bool has_supplied_multimodal_options =
        !options.mmproj.empty() || options.soft_token_buffer_set || !options.mmproj_offload ||
        options.image_min_tokens != -1 || options.image_max_tokens != -1 || options.mtmd_batch_max_tokens != 1024 ||
        !options.media_path.empty() || options.media_max_bytes != 64u * 1024u * 1024u ||
        options.multimodal_scope != MOE_MEASURE_INPUT_SCOPE_MEDIA;
    if (!has_media && has_supplied_multimodal_options) {
        LOG_INF("moe-measure: datasets contain no media; multimodal options are inactive\n");
    }
    LOG_INF("moe-measure: reading model structure\n");

    moe_measure_model_info model_info;
    if (!moe_measure_model_info_load(params.model.path, model_info, error)) {
        LOG_ERR("error: %s\n", error.c_str());
        return 1;
    }
    if (model_info.architecture == "llama4") {
        LOG_ERR(
            "error: Llama 4 applies router weights before the expert FFN and is not supported by this MoE measurement metric\n");
        return 1;
    }

    std::vector<uint32_t> moe_expert_counts;
    for (int32_t layer : model_info.moe_layers) {
        moe_expert_counts.push_back(model_info.n_expert_per_layer[layer]);
    }
    const auto [min_experts, max_experts] = std::minmax_element(moe_expert_counts.begin(), moe_expert_counts.end());
    if (*min_experts == *max_experts) {
        LOG_INF("moe-measure: architecture %s, %u layers (%zu MoE), %u experts, top-%u routing\n",
                model_info.architecture.c_str(), model_info.n_layer, model_info.moe_layers.size(), *max_experts,
                model_info.n_expert_used);
    } else {
        LOG_INF("moe-measure: architecture %s, %u layers (%zu MoE), %u..%u experts, top-%u routing\n",
                model_info.architecture.c_str(), model_info.n_layer, model_info.moe_layers.size(), *min_experts,
                *max_experts, model_info.n_expert_used);
    }
    llama_backend_init();
    llama_numa_init(params.numa);
    LOG_INF("moe-measure: loading vocabulary-only model for preprocessing\n");
    llama_model_params vocab_model_params = common_model_params_to_llama(params);
    vocab_model_params.vocab_only         = true;
    llama_model_ptr vocab_model(llama_model_load_from_file(params.model.path.c_str(), vocab_model_params));
    if (!vocab_model) {
        LOG_ERR("error: failed to initialize vocabulary-only model\n");
        return 1;
    }
    const size_t        context_size = static_cast<size_t>(requested_context_size);
    const llama_vocab * vocab        = llama_model_get_vocab(vocab_model.get());
    const size_t        model_n_embd = model_info.n_embd;
    moe_measure_vocab_mask     vocab_mask;
    if (!build_vocab_mask(options, vocab, vocab_mask, error)) {
        LOG_ERR("error: %s\n", error.c_str());
        return 1;
    }
    LOG_INF("moe-measure: vocabulary mask selects %zu of %d token IDs (hash %016llx)\n",
            static_cast<size_t>(std::count(vocab_mask.selected.begin(), vocab_mask.selected.end(), uint8_t(1))),
            llama_vocab_n_tokens(vocab), (unsigned long long) vocab_mask.hash);
    common_chat_templates_ptr templates;
    try {
        templates = common_chat_templates_init(vocab_model.get(), params.chat_template);
    } catch (const std::exception & exception) {
        LOG_ERR("error: failed to initialize chat template: %s\n", exception.what());
        return 1;
    }
    LOG_INF("moe-measure: chat template ready (%zu bytes)\n", common_chat_templates_source(templates.get()).size());

    constexpr uint32_t media_pipeline_version = 1;
    const std::string  media_marker           = mtmd_default_marker();
    uint64_t           projector_signature    = 0;
    if (has_media) {
        LOG_INF("moe-measure: reading projector structure: %s\n", options.mmproj.c_str());
        if (!moe_measure_gguf_structural_signature(options.mmproj, projector_signature, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
    }

    moe_measure_measurement_header header;
    header.architecture    = model_info.architecture;
    header.model_signature = model_info.signature;
    header.template_hash   = moe_measure_hash_string(0, common_chat_templates_source(templates.get()));
    header.template_hash   = moe_measure_hash_bytes(header.template_hash, &params.use_jinja, sizeof(params.use_jinja));
    header.template_hash =
        moe_measure_hash_bytes(header.template_hash, &params.enable_reasoning, sizeof(params.enable_reasoning));
    for (const auto & item : params.default_template_kwargs) {
        header.template_hash = moe_measure_hash_string(header.template_hash, item.first);
        header.template_hash = moe_measure_hash_string(header.template_hash, item.second);
    }
    header.tokenization_hash         = hash_vocab(vocab);
    header.n_layer                   = model_info.n_layer;
    header.n_expert                  = model_info.n_expert;
    header.n_expert_used             = model_info.n_expert_used;
    header.n_ctx                     = static_cast<uint32_t>(requested_context_size);
    header.router_weights_normalized = options.normalize_router_weights;
    header.projector_signature       = projector_signature;
    header.media_pipeline_version    = has_media ? media_pipeline_version : 0;
    header.input_scope =
        moe_measure_effective_input_scope(has_media, inventory.image_paired_records > 0, options.multimodal_scope);
    header.output_role     = MOE_MEASURE_OUTPUT_ROLE_PRIMARY;
    header.vocab_mask_hash = vocab_mask.hash;
    if (inventory.image_paired_records > 0 && options.multimodal_scope != MOE_MEASURE_INPUT_SCOPE_ALL) {
        LOG_INF("moe-measure: paired input records select owned text and media positions; measurement scope is all\n");
    }
    if (has_media) {
        header.media_config_hash = moe_measure_hash_bytes(0, &media_pipeline_version, sizeof(media_pipeline_version));
        header.media_config_hash = moe_measure_hash_string(header.media_config_hash, media_marker);
        header.media_config_hash =
            moe_measure_hash_bytes(header.media_config_hash, &options.image_min_tokens, sizeof(options.image_min_tokens));
        header.media_config_hash =
            moe_measure_hash_bytes(header.media_config_hash, &options.image_max_tokens, sizeof(options.image_max_tokens));
        header.media_config_hash = moe_measure_hash_bytes(header.media_config_hash, &options.mtmd_batch_max_tokens,
                                                   sizeof(options.mtmd_batch_max_tokens));
        header.media_config_hash =
            moe_measure_hash_bytes(header.media_config_hash, &options.media_max_bytes, sizeof(options.media_max_bytes));
    }
    header.moe_layers = model_info.moe_layers;

    moe_measure_measurement_summary summary;
    if (std::ifstream(params.out_file, std::ios::binary).good()) {
        if (!reconcile_existing_measurement(params.out_file, "measurement", has_media, options, header, summary,
                                            error)) {
            if (!error.empty()) {
                LOG_ERR("error: %s\n", error.c_str());
            }
            return 1;
        }
        if (!moe_measure_measurement_truncate_tail(params.out_file, summary, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
        header = summary.header;
        LOG_INF("moe-measure: resuming %s after %llu completed contexts\n", params.out_file.c_str(),
                (unsigned long long) summary.n_blocks);
    } else {
        LOG_INF("moe-measure: creating measurement log %s\n", params.out_file.c_str());
        header.measurement_id = moe_measure_make_id(params.out_file);
        if (!moe_measure_measurement_create(params.out_file, header, error) ||
            !moe_measure_measurement_read(params.out_file, summary, {}, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
    }
    collector.init(header, moe_expert_counts, options.collector);

    moe_measure_measure_output primary_output;
    primary_output.path    = params.out_file;
    primary_output.header  = header;
    primary_output.summary = summary;
    std::unique_ptr<moe_measure_measure_output> prefix_output;
    if (!options.prefix_output.empty()) {
        prefix_output                         = std::make_unique<moe_measure_measure_output>();
        prefix_output->path                   = options.prefix_output;
        prefix_output->header                 = header;
        prefix_output->header.format_version  = header.format_version;
        prefix_output->header.output_role     = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
        prefix_output->header.vocab_mask_hash = vocab_mask.hash;
        if (std::ifstream(prefix_output->path, std::ios::binary).good()) {
            if (!reconcile_existing_measurement(prefix_output->path, "prefix measurement", has_media, options,
                                                prefix_output->header, prefix_output->summary, error)) {
                if (!error.empty()) {
                    LOG_ERR("error: %s\n", error.c_str());
                }
                return 1;
            }
            if (!moe_measure_measurement_truncate_tail(prefix_output->path, prefix_output->summary, error)) {
                LOG_ERR("error: %s\n", error.c_str());
                return 1;
            }
            prefix_output->header = prefix_output->summary.header;
        } else {
            prefix_output->header.measurement_id = moe_measure_make_id(prefix_output->path);
            if (!moe_measure_measurement_create(prefix_output->path, prefix_output->header, error) ||
                !moe_measure_measurement_read(prefix_output->path, prefix_output->summary, {}, error)) {
                LOG_ERR("error: %s\n", error.c_str());
                return 1;
            }
        }
    }

    uint64_t skipped = 0;

    std::string concatenated_text;
    for (size_t i = 0; i < options.text_files.size(); ++i) {
        LOG_INF("moe-measure: reading text dataset %zu of %zu: %s\n", i + 1, options.text_files.size(),
                options.text_files[i].c_str());
        std::string current;
        if (!read_file(options.text_files[i], current, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
        if (i > 0) {
            concatenated_text.push_back('\n');
        }
        concatenated_text += current;
    }
    LOG_INF("moe-measure: tokenizing %zu concatenated text bytes\n", concatenated_text.size());
    std::vector<llama_token> text_tokens;
    if (!concatenated_text.empty()) {
        text_tokens = common_tokenize(vocab, concatenated_text, true, params.parse_special);
    }

    size_t soft_token_buffer_bytes = 0;
    if (has_media && options.soft_token_buffer_set) {
        if (!gib_to_bytes(options.soft_token_buffer_gib, soft_token_buffer_bytes)) {
            LOG_ERR("error: --soft-token-buffer-gib is too large\n");
            return 1;
        }
    } else if (has_media) {
        soft_token_buffer_bytes = default_soft_token_buffer_bytes();
    }
    if (has_media) {
        LOG_INF("moe-measure: soft-token buffer: %.2f GiB\n",
                soft_token_buffer_bytes / (1024.0 * 1024.0 * 1024.0));
    }

    mtmd::context_ptr mctx;
    auto              load_projector = [&](std::string & load_error) -> mtmd_context * {
        if (mctx) {
            return mctx.get();
        }
        if (!has_media) {
            load_error = "image-bearing record appeared after the text-only dataset inventory";
            return nullptr;
        }
        if (options.mmproj.empty()) {
            load_error = "image-bearing record requires --mmproj";
            return nullptr;
        }
        mtmd_context_params mparams = mtmd_context_params_default();
        mparams.use_gpu             = options.mmproj_offload;
        mparams.print_timings       = true;
        mparams.n_threads           = params.cpuparams.n_threads;
        mparams.media_marker        = media_marker.c_str();
        mparams.flash_attn_type     = params.flash_attn_type;
        mparams.warmup              = false;
        mparams.image_min_tokens    = options.image_min_tokens;
        mparams.image_max_tokens    = options.image_max_tokens;
        mparams.batch_max_tokens    = options.mtmd_batch_max_tokens;
        LOG_INF("moe-measure: loading multimodal projector on %s\n", options.mmproj_offload ? "GPU" : "CPU");
        mctx.reset(mtmd_init_from_file(options.mmproj.c_str(), vocab_model.get(), mparams));
        if (!mctx || !mtmd_support_vision(mctx.get())) {
            mctx.reset();
            load_error = "failed to initialize a vision-capable projector from " + options.mmproj;
            return nullptr;
        }
        return mctx.get();
    };

    std::unordered_set<uint64_t> reserved_hashes = summary.context_hashes;
    std::unordered_set<uint64_t> prefix_reserved_hashes =
        prefix_output ? prefix_output->summary.context_hashes : std::unordered_set<uint64_t>();
    moe_measure_task_source source(options, vocab, templates.get(), params, load_projector, collector.header,
                            prefix_output ? &prefix_output->header : nullptr, vocab_mask, context_size,
                            std::move(text_tokens), reserved_hashes, prefix_output ? &prefix_reserved_hashes : nullptr,
                            skipped);
    source.set_text_source_id(moe_measure_hash_string(moe_measure_hash_string(0, "text"), concatenated_text));

    bool                               source_done = false;
    size_t                             cycle       = 0;
    std::unique_ptr<moe_measure_measure_task> deferred;
    while (!source_done || deferred) {
        cycle++;
        std::vector<std::unique_ptr<moe_measure_measure_task>> tasks;
        size_t                                          soft_bytes = 0;

        while (!source_done) {
            std::unique_ptr<moe_measure_measure_task> task;
            if (deferred) {
                task = std::move(deferred);
            } else {
                bool       done              = false;
                const auto preparation_start = moe_measure_clock::now();
                if (!source.next(task, done, error)) {
                    LOG_ERR("error: %s\n", error.c_str());
                    return 1;
                }
                collector.perf.preparation_ms += elapsed_ms(preparation_start);
                if (done) {
                    source_done = true;
                    break;
                }
            }
            if (!task) {
                continue;
            }
            if (task->multimodal) {
                mtmd_context * projector = load_projector(error);
                if (projector == nullptr) {
                    LOG_ERR("error: %s\n", error.c_str());
                    return 1;
                }
                size_t        task_soft_bytes = 0;
                if (model_n_embd == 0 || !task_soft_token_bytes(*task, model_n_embd, task_soft_bytes)) {
                    LOG_ERR("error: soft-token buffer size overflow\n");
                    return 1;
                }
                if (soft_bytes > 0 &&
                    (soft_bytes >= soft_token_buffer_bytes || task_soft_bytes > soft_token_buffer_bytes - soft_bytes)) {
                    deferred = std::move(task);
                    break;
                }
                if (task_soft_bytes > soft_token_buffer_bytes) {
                    LOG_WRN("moe-measure: context %016llx requires %.2f GiB of soft tokens and will run alone\n",
                            (unsigned long long) task->context_hash, task_soft_bytes / (1024.0 * 1024.0 * 1024.0));
                }
                const auto preparation_start = moe_measure_clock::now();
                if (!materialize_multimodal_task(*task, projector, model_n_embd, collector, error)) {
                    LOG_ERR("error: %s\n", error.c_str());
                    return 1;
                }
                collector.perf.preparation_ms += elapsed_ms(preparation_start);
                soft_bytes += task_soft_bytes;
            }
            tasks.push_back(std::move(task));
        }

        if (mctx) {
            LOG_INF("moe-measure: unloading multimodal projector after preparing %.2f GiB of soft tokens\n",
                    soft_bytes / (1024.0 * 1024.0 * 1024.0));
            mctx.reset();
        }
        if (tasks.empty()) {
            if (source_done) {
                LOG_INF("moe-measure: no pending contexts; full model load avoided\n");
                break;
            }
            continue;
        }

        LOG_INF("moe-measure: cycle %zu prepared %zu contexts; loading language model and %d-token aggregate context\n", cycle,
                tasks.size(), params.n_ctx);
        common_init_result_ptr init  = common_init_from_params(params);
        llama_model *          model = init ? init->model() : nullptr;
        llama_context *        ctx   = init ? init->context() : nullptr;
        if (model == nullptr || ctx == nullptr) {
            LOG_ERR("error: failed to initialize full model\n");
            return 1;
        }
        const size_t active_context_size = llama_n_ctx_seq(ctx);
        if (llama_n_seq_max(ctx) != static_cast<uint32_t>(params.n_parallel) || active_context_size < context_size) {
            LOG_ERR("error: initialized context does not provide the requested per-slot capacity\n");
            return 1;
        }
        const size_t decode_capacity = std::max<size_t>(
            1, std::min({ static_cast<uint32_t>(params.n_batch), llama_n_ubatch(ctx), llama_n_batch(ctx) }));
        LOG_INF("moe-measure: active per-slot context: %zu, decode capacity: %zu (-b %d, -ub %d)\n", active_context_size,
                decode_capacity, params.n_batch, params.n_ubatch);
        if (!run_parallel_measurements(tasks, ctx, collector, primary_output, prefix_output.get(),
                                       static_cast<size_t>(params.n_parallel), decode_capacity, error)) {
            LOG_ERR("error: %s\n", error.c_str());
            return 1;
        }
        llama_perf_context_print(ctx);
        LOG_INF("moe-measure: unloading language model after cycle %zu\n", cycle);
        init.reset();
    }

    LOG_INF("moe-measure: measurement complete: %llu primary and %llu prefix contexts appended, %llu already present\n",
            (unsigned long long) primary_output.measured,
            (unsigned long long) (prefix_output ? prefix_output->measured : 0), (unsigned long long) skipped);
    const moe_measure_performance & perf    = collector.perf;
    const double measured_per_second = perf.decode_ms > 0.0 ? 1000.0 * perf.measured_tokens / perf.decode_ms : 0.0;
    LOG_INF("moe-measure: performance: %.2f measured tokens/s, %llu decoded, %llu measured, %llu callbacks\n",
            measured_per_second, (unsigned long long) perf.decoded_tokens, (unsigned long long) perf.measured_tokens,
            (unsigned long long) perf.callback_calls);
    LOG_INF(
        "moe-measure: timings: decode %.1f ms, collector copies %.1f ms, CPU scoring %.1f ms, preparation %.1f ms, "
        "projector %.1f ms, commits %.1f ms; callback transfer %llu bytes (%.2f MiB)\n",
        perf.decode_ms, perf.copy_ms, perf.scoring_ms, perf.preparation_ms, perf.projector_ms, perf.commit_ms,
        (unsigned long long) perf.callback_bytes, perf.callback_bytes / (1024.0 * 1024.0));
    vocab_model.reset();
    llama_backend_free();
    return 0;
}
