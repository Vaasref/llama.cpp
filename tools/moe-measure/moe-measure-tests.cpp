#include "moe-measure-scheduler.h"

#include "common.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"

#include <cmath>
#include <iostream>

namespace {

void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::cerr << "test failure at line " << line << ": " << expression << '\n';
        exit(1);
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

}  // namespace

int main() {
    const llama_model_params default_model_params = llama_model_default_params();
    REQUIRE(!default_model_params.no_output);

    common_params capture_params;
    capture_params.expert_output_capture_only = true;
    REQUIRE(common_model_params_to_llama(capture_params).no_output);

    const llama_context_params default_context_params = llama_context_default_params();
    REQUIRE(!default_context_params.expert_output_capture);
    REQUIRE(!default_context_params.expert_output_capture_only);
    REQUIRE(default_context_params.cb_eval_row_order == nullptr);

    int32_t total = 0;
    REQUIRE(moe_measure_expand_context_size(4096, 4, total));
    REQUIRE(total == 16384);
    REQUIRE(!moe_measure_expand_context_size(4096, 0, total));
    REQUIRE(!moe_measure_expand_context_size(std::numeric_limits<int32_t>::max(), 2, total));

    REQUIRE(moe_measure_allocate_slot_tokens({ 7, 3, 1 }, 6) == std::vector<size_t>({ 3, 2, 1 }));
    REQUIRE(moe_measure_allocate_slot_tokens({ 7, 3, 1 }, 20) == std::vector<size_t>({ 7, 3, 1 }));
    REQUIRE(moe_measure_allocate_slot_tokens({ 0, 4, 4, 4 }, 2) == std::vector<size_t>({ 0, 1, 1, 0 }));

    std::vector<size_t> layer_offsets = { 0, 0 };
    size_t              row_begin     = 0;
    REQUIRE(moe_measure_consume_layer_rows(layer_offsets, 0, 3, 5, row_begin) && row_begin == 0);
    REQUIRE(moe_measure_consume_layer_rows(layer_offsets, 0, 2, 5, row_begin) && row_begin == 3);
    REQUIRE(!moe_measure_consume_layer_rows(layer_offsets, 0, 1, 5, row_begin));

    const int32_t row_order[] = { 0, 3, 4, 1, 2 };
    REQUIRE(moe_measure_validate_row_order(row_order, 5, 5));
    const int32_t duplicate_row_order[] = { 0, 1, 1 };
    REQUIRE(!moe_measure_validate_row_order(duplicate_row_order, 3, 3));
    std::vector<uint8_t> rows_seen(5, 0);
    const int32_t first_row_chunk[]  = { 0, 3, 4 };
    const int32_t second_row_chunk[] = { 1, 2 };
    REQUIRE(moe_measure_append_row_order(first_row_chunk, 3, 5, rows_seen));
    REQUIRE(moe_measure_append_row_order(second_row_chunk, 2, 5, rows_seen));
    REQUIRE(!moe_measure_append_row_order(second_row_chunk, 2, 5, rows_seen));

    int32_t captured_expert = -1;
    REQUIRE(moe_measure_decode_captured_expert(0.0f, 128, captured_expert) && captured_expert == 0);
    REQUIRE(moe_measure_decode_captured_expert(127.0f, 128, captured_expert) && captured_expert == 127);
    REQUIRE(!moe_measure_decode_captured_expert(-1.0f, 128, captured_expert));
    REQUIRE(!moe_measure_decode_captured_expert(128.0f, 128, captured_expert));
    REQUIRE(!moe_measure_decode_captured_expert(1.5f, 128, captured_expert));
    REQUIRE(!moe_measure_decode_captured_expert(std::numeric_limits<float>::infinity(), 128, captured_expert));
    REQUIRE(!moe_measure_decode_captured_expert(std::numeric_limits<float>::quiet_NaN(), 128, captured_expert));

    moe_measure_measurement_header header;
    header.n_expert_used = 2;
    header.moe_layers    = { 1, 3 };
    moe_measure_measure_task task;
    task.tokens   = { 1, 2, 3 };
    task.segments = {
        { moe_measure_segment_kind::text, moe_measure_measure_destination::primary, 0, 0, 3 }
    };
    moe_measure_init_measurement_block(task, header, 11, 22, task.tokens.size());
    REQUIRE(task.block.n_tokens == 3);
    REQUIRE(task.block.token_ids.size() == 3);
    REQUIRE(task.block.expert_ids.empty());
    REQUIRE(task.block.contributions.empty());
    moe_measure_prepare_measurement_observations(task.block, header);
    REQUIRE(task.block.expert_ids.size() == 12);
    REQUIRE(task.block.contributions.size() == 12);

    size_t soft_bytes = 0;
    REQUIRE(moe_measure_soft_token_bytes(1024, 4096, soft_bytes));
    REQUIRE(soft_bytes == size_t(1024) * 4096 * sizeof(float));
    REQUIRE(!moe_measure_soft_token_bytes(std::numeric_limits<size_t>::max(), 2, soft_bytes));
    REQUIRE(moe_measure_resolve_position(3, 4, 7) == 10);
    REQUIRE(moe_measure_resolve_position(5, 5, 100) == 5);

    moe_measure_measure_slot slot;
    slot.task           = std::make_unique<moe_measure_measure_task>();
    slot.task->tokens   = { 4, 5, 6, 7 };
    slot.task->segments = {
        { moe_measure_segment_kind::text, moe_measure_measure_destination::primary, 0, 0, 4 }
    };
    REQUIRE(moe_measure_slot_segment_tokens(slot) == 4);

    slot.task->token_mask = { 1, 0, 1, 0 };
    slot.task->segments   = {
        { moe_measure_segment_kind::text, moe_measure_measure_destination::prefix, 0, 1, 2 }
    };
    REQUIRE(moe_measure_slot_segment_tokens(slot) == 2);
    REQUIRE(!moe_measure_slot_text_token_selected(slot, 0));
    REQUIRE(moe_measure_slot_text_token_selected(slot, 1));

    slot.task->multimodal = true;
    slot.task->prepared_chunks.resize(1);
    slot.task->prepared_chunks[0].kind   = moe_measure_segment_kind::media;
    slot.task->prepared_chunks[0].n_embd = 2;
    slot.task->prepared_chunks[0].n_pos  = 3;
    slot.task->segments                  = {
        { moe_measure_segment_kind::media, moe_measure_measure_destination::primary, 0, 0, 2 }
    };
    REQUIRE(moe_measure_slot_prepared_chunk(slot) == &slot.task->prepared_chunks[0]);
    REQUIRE(moe_measure_slot_prepared_chunk(slot)->n_embd == 2);

    moe_measure_measurement_block prefix_block;
    moe_measure_init_measurement_block(prefix_block, header, 33, 44, 2);
    REQUIRE(prefix_block.context_hash == 44);
    REQUIRE(prefix_block.n_tokens == 2);
    REQUIRE(prefix_block.token_ids.size() == 2);
    REQUIRE(prefix_block.expert_ids.size() == 8);

    std::vector<moe_measure_measure_slot> mixed_slots(3);
    for (size_t i = 0; i < mixed_slots.size(); ++i) {
        mixed_slots[i].seq_id = i;
        mixed_slots[i].task   = std::make_unique<moe_measure_measure_task>();
        mixed_slots[i].task->segments.push_back({ i == 1 ? moe_measure_segment_kind::media : moe_measure_segment_kind::text,
                                                  moe_measure_measure_destination::primary, 0, 0, 1 });
    }
    const std::vector<moe_measure_measure_slot *> text_slots = moe_measure_select_text_slots(mixed_slots);
    REQUIRE(text_slots.size() == 2);
    REQUIRE(text_slots[0]->seq_id == 0 && text_slots[1]->seq_id == 2);

    std::vector<moe_measure_measure_slot> fill_slots(3);
    fill_slots[1].task = std::make_unique<moe_measure_measure_task>();
    REQUIRE(moe_measure_select_empty_slots(fill_slots, 0).empty());
    const std::vector<moe_measure_measure_slot *> one_empty_slot = moe_measure_select_empty_slots(fill_slots, 1);
    REQUIRE(one_empty_slot.size() == 1 && one_empty_slot[0] == &fill_slots[0]);
    const std::vector<moe_measure_measure_slot *> all_empty_slots = moe_measure_select_empty_slots(fill_slots, 10);
    REQUIRE(all_empty_slots.size() == 2);
    REQUIRE(all_empty_slots[0] == &fill_slots[0] && all_empty_slots[1] == &fill_slots[2]);

    ggml_init_params graph_params = { 1024 * 1024, nullptr, true };
    ggml_context *   graph_ctx    = ggml_init(graph_params);
    REQUIRE(graph_ctx != nullptr);
    ggml_tensor * graph_a = ggml_new_tensor_1d(graph_ctx, GGML_TYPE_F32, 4);
    ggml_tensor * graph_b = ggml_sqr(graph_ctx, graph_a);
    ggml_tensor * graph_c = ggml_sqrt(graph_ctx, graph_b);
    ggml_cgraph * graph   = ggml_new_graph(graph_ctx);
    ggml_build_forward_expand(graph, graph_c);
    REQUIRE(ggml_graph_trim_after(graph, graph_b));
    REQUIRE(ggml_graph_node(graph, -1) == graph_b);

    ggml_tensor * expert_rows = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32, 8, 2, 12);
    ggml_tensor * weights     = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_F32, 1, 2, 12);
    ggml_tensor * ids         = ggml_new_tensor_3d(graph_ctx, GGML_TYPE_I32, 1, 2, 12);
    ggml_tensor * norm2       = ggml_sum_rows(graph_ctx, ggml_sqr(graph_ctx, expert_rows));
    ggml_tensor * ids_f32     = ggml_cast(graph_ctx, ids, GGML_TYPE_F32);
    ggml_tensor * capture     = ggml_concat(graph_ctx, ggml_concat(graph_ctx, norm2, weights, 0), ids_f32, 0);
    REQUIRE(capture->type == GGML_TYPE_F32);
    REQUIRE(capture->ne[0] == 3);
    REQUIRE(capture->ne[1] == 2);
    REQUIRE(capture->ne[2] == 12);
    REQUIRE(capture->ne[3] == 1);
    ggml_set_output(capture);
    ggml_build_forward_expand(graph, capture);

    ggml_backend_t backend = ggml_backend_cpu_init();
    REQUIRE(backend != nullptr);
    ggml_backend_buffer_t buffer = ggml_backend_alloc_ctx_tensors(graph_ctx, backend);
    REQUIRE(buffer != nullptr);
    std::vector<float> expert_values(8 * 2 * 12);
    std::vector<float> weight_values(2 * 12);
    std::vector<int32_t> id_values(2 * 12);
    for (size_t row = 0; row < 12; ++row) {
        for (size_t route = 0; route < 2; ++route) {
            weight_values[route + 2 * row] = 0.25f * (route + 1);
            id_values[route + 2 * row]     = static_cast<int32_t>(10 + route + 2 * row);
            for (size_t value = 0; value < 8; ++value) {
                expert_values[value + 8 * (route + 2 * row)] = static_cast<float>(value + 1);
            }
        }
    }
    ggml_backend_tensor_set(expert_rows, expert_values.data(), 0, expert_values.size() * sizeof(float));
    ggml_backend_tensor_set(weights, weight_values.data(), 0, weight_values.size() * sizeof(float));
    ggml_backend_tensor_set(ids, id_values.data(), 0, id_values.size() * sizeof(int32_t));
    REQUIRE(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS);
    std::vector<float> capture_values(3 * 2 * 12);
    ggml_backend_tensor_get(capture, capture_values.data(), 0, capture_values.size() * sizeof(float));
    const float expected_norm2 = 204.0f;
    for (size_t row = 0; row < 12; ++row) {
        for (size_t route = 0; route < 2; ++route) {
            const size_t base = 3 * (route + 2 * row);
            REQUIRE(std::fabs(capture_values[base] - expected_norm2) < 1e-6f);
            REQUIRE(capture_values[base + 1] == weight_values[route + 2 * row]);
            REQUIRE(capture_values[base + 2] == static_cast<float>(id_values[route + 2 * row]));
        }
    }
    ggml_backend_buffer_free(buffer);
    ggml_backend_free(backend);
    ggml_free(graph_ctx);

    std::cout << "MoE measurement scheduler tests passed\n";
    return 0;
}
