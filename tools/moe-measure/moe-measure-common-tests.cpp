#include "moe-measure-common.h"

#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::cerr << "test failure at line " << line << ": " << expression << '\n';
        exit(1);
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

moe_measure_measurement_header header(moe_measure_id id) {
    moe_measure_measurement_header result;
    result.architecture      = "test";
    result.model_signature   = 0x1234;
    result.measurement_id    = id;
    result.template_hash     = 0x5678;
    result.tokenization_hash = 0x9abc;
    result.n_layer           = 1;
    result.n_expert          = 4;
    result.n_expert_used     = 2;
    result.n_ctx             = 16;
    result.moe_layers        = { 0 };
    return result;
}

moe_measure_measurement_block block(uint64_t context, uint32_t a, float av, uint32_t b, float bv) {
    moe_measure_measurement_block result;
    result.source_id     = 9;
    result.context_hash  = context;
    result.n_tokens      = 1;
    result.token_hashes  = { context + 1 };
    result.token_ids     = { static_cast<int32_t>(context + 2) };
    result.expert_ids    = { a, b };
    result.contributions = { av, bv };
    return result;
}

bool close(float a, float b) {
    return std::fabs(a - b) < 1e-5f;
}

constexpr std::array<uint8_t, 224> LEGACY_MEASUREMENT = {
    0x4c, 0x52, 0x45, 0x41, 0x50, 0x4d, 0x31, 0x00, 0x05, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x8c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x34, 0x12, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x78, 0x56, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbc, 0x9a, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x10, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
    0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x74, 0x65, 0x73, 0x74, 0x00, 0x00, 0x00, 0x00, 0x52, 0x45, 0x50, 0x4d,
    0x42, 0x4c, 0x4b, 0x31, 0x3c, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xd2, 0x1c, 0x9b, 0x79,
    0x26, 0xf1, 0x9a, 0x10, 0x09, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0a, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x0b, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x0c, 0x00, 0x00, 0x00,
    0x00, 0x00, 0x00, 0x00, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x40, 0x00, 0x00, 0x80, 0x40,
};

void write_bytes(const std::filesystem::path & path, const uint8_t * data, size_t size) {
    std::ofstream output(path, std::ios::binary);
    output.write(reinterpret_cast<const char *>(data), size);
    REQUIRE(output.good());
}

}  // namespace

int main() {
    REQUIRE(moe_measure_hash_bytes(0, "abc", 3) == 0xe71fa2190541574bULL);
    moe_measure_measurement_header requested_header = header({ 1, 2 });
    moe_measure_measurement_header existing_header  = requested_header;
    existing_header.n_ctx                    = 32;
    existing_header.input_scope              = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    existing_header.moe_layers               = { 1, 3 };
    const std::vector<moe_measure_measurement_header_mismatch> header_mismatches =
        moe_measure_measurement_header_mismatches(requested_header, existing_header);
    REQUIRE(header_mismatches.size() == 3);
    REQUIRE(header_mismatches[0].field == "n_ctx");
    REQUIRE(header_mismatches[0].requested == "16");
    REQUIRE(header_mismatches[0].existing == "32");
    REQUIRE(header_mismatches[1].field == "input_scope");
    REQUIRE(header_mismatches[1].requested == "text");
    REQUIRE(header_mismatches[1].existing == "media");
    REQUIRE(header_mismatches[2].field == "moe_layers");
    REQUIRE(header_mismatches[2].requested == "[0]");
    REQUIRE(header_mismatches[2].existing == "[1,3]");
    existing_header                    = requested_header;
    existing_header.projector_signature = 9;
    existing_header.media_config_hash   = 10;
    existing_header.input_scope         = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    REQUIRE(moe_measure_measurement_header_mismatches(requested_header, existing_header, false).empty());
    existing_header.vocab_mask_hash = 11;
    const std::vector<moe_measure_measurement_header_mismatch> text_only_mismatches =
        moe_measure_measurement_header_mismatches(requested_header, existing_header, false);
    REQUIRE(text_only_mismatches.size() == 1);
    REQUIRE(text_only_mismatches[0].field == "vocab_mask_hash");
    existing_header = requested_header;
    existing_header.model_signature ^= 1;
    const std::vector<moe_measure_measurement_header_mismatch> hash_mismatches =
        moe_measure_measurement_header_mismatches(requested_header, existing_header);
    REQUIRE(hash_mismatches.size() == 1);
    REQUIRE(hash_mismatches[0].requested == "0x0000000000001234");
    REQUIRE(hash_mismatches[0].existing == "0x0000000000001235");

    float contribution = 0.0f;
    REQUIRE(moe_measure_calculate_contribution(0.25, 0.5, 25.0, true, contribution));
    REQUIRE(close(contribution, 2.5f));
    REQUIRE(moe_measure_calculate_contribution(0.25, 0.5, 25.0, false, contribution));
    REQUIRE(close(contribution, 1.25f));
    REQUIRE(!moe_measure_calculate_contribution(0.25, 0.0, 25.0, true, contribution));

    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("llama-moe-measure-test-" + moe_measure_id_string(moe_measure_make_id("test")));
    std::filesystem::create_directories(root);
    const std::string include_path = (root / "include.reapm").string();
    const std::string exclude_path = (root / "exclude.moem").string();
    std::string       error;

    write_bytes(include_path, LEGACY_MEASUREMENT.data(), LEGACY_MEASUREMENT.size());
    moe_measure_measurement_summary legacy_summary;
    REQUIRE(moe_measure_measurement_read(include_path, legacy_summary, {}, error));
    REQUIRE(legacy_summary.header.format_version == 5);
    REQUIRE(legacy_summary.n_blocks == 1);
    REQUIRE(legacy_summary.context_hashes.count(10) == 1);

    std::array<uint8_t, LEGACY_MEASUREMENT.size()> invalid_layers = LEGACY_MEASUREMENT;
    invalid_layers[64] = 0x01;
    invalid_layers[65] = 0x02;
    const std::string invalid_layers_path = (root / "invalid-layers.moem").string();
    write_bytes(invalid_layers_path, invalid_layers.data(), invalid_layers.size());
    REQUIRE(!moe_measure_measurement_read(invalid_layers_path, legacy_summary, {}, error));

    std::array<uint8_t, LEGACY_MEASUREMENT.size()> oversized_block = LEGACY_MEASUREMENT;
    oversized_block[148] = 0xff;
    oversized_block[149] = 0xff;
    const std::string oversized_block_path = (root / "oversized-block.moem").string();
    write_bytes(oversized_block_path, oversized_block.data(), oversized_block.size());
    REQUIRE(!moe_measure_measurement_read(oversized_block_path, legacy_summary, {}, error));

    moe_measure_measurement_header invalid_header = header({ 0, 0 });
    invalid_header.output_role             = MOE_MEASURE_OUTPUT_ROLE_UNKNOWN;
    REQUIRE(!moe_measure_measurement_create((root / "invalid.moem").string(), invalid_header, error));

    moe_measure_measurement_header include_header = header({ 1, 2 });
    include_header.projector_signature     = 0x2222;
    include_header.media_config_hash       = 0x3333;
    include_header.media_pipeline_version  = 1;
    include_header.input_scope             = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    include_header.output_role             = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
    include_header.vocab_mask_hash         = 0x4444;
    const std::string generated_path = (root / "generated.moem").string();
    REQUIRE(moe_measure_measurement_create(generated_path, include_header, error));
    moe_measure_measurement_summary summary;
    REQUIRE(moe_measure_measurement_read(generated_path, summary, {}, error));
    REQUIRE(summary.valid_size == std::filesystem::file_size(generated_path));
    moe_measure_measurement_block invalid_token = block(9, 0, 1.0f, 1, 1.0f);
    invalid_token.token_ids[0]            = -2;
    REQUIRE(!moe_measure_measurement_append(generated_path, include_header, invalid_token, error));
    REQUIRE(moe_measure_measurement_append(generated_path, include_header, block(10, 0, 2.0f, 1, 4.0f), error));
    size_t callbacks = 0;
    REQUIRE(moe_measure_measurement_read(
        generated_path, summary,
        [&](const moe_measure_measurement_block & value) {
            callbacks++;
            return value.context_hash == 10 && value.token_ids[0] == 12 && value.expert_ids[1] == 1;
        },
        error));
    REQUIRE(callbacks == 1 && summary.n_blocks == 1 && summary.context_hashes.count(10) == 1 &&
            !summary.has_partial_tail);
    REQUIRE(summary.has_text_tokens);
    REQUIRE(!summary.has_media_tokens);
    REQUIRE(summary.header.format_version == 5);

    moe_measure_measurement_header media_header = header(moe_measure_make_id("media-kind"));
    media_header.input_scope             = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    const std::string media_path         = (root / "media-kind.moem").string();
    REQUIRE(moe_measure_measurement_create(media_path, media_header, error));
    moe_measure_measurement_block media_block = block(12, 0, 1.0f, 1, 1.0f);
    media_block.token_ids[0]           = -1;
    REQUIRE(moe_measure_measurement_append(media_path, media_header, media_block, error));
    REQUIRE(moe_measure_measurement_read(media_path, summary, {}, error));
    REQUIRE(!summary.has_text_tokens);
    REQUIRE(summary.has_media_tokens);

    moe_measure_measurement_header prefix_header = include_header;
    prefix_header.output_role             = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
    prefix_header.measurement_id          = moe_measure_make_id("prefix-v5");
    const std::string prefix_path          = (root / "prefix-v5.moem").string();
    REQUIRE(moe_measure_measurement_create(prefix_path, prefix_header, error));
    REQUIRE(moe_measure_measurement_append(prefix_path, prefix_header, block(20, 1, 3.0f, 2, 5.0f), error));
    bool prefix_seen = false;
    REQUIRE(moe_measure_measurement_read(
        prefix_path, summary,
        [&](const moe_measure_measurement_block & value) {
            prefix_seen = value.token_ids.size() == 1 && value.token_ids[0] == 22;
            return true;
        },
        error));
    REQUIRE(prefix_seen && summary.header.format_version == 5 &&
            summary.header.output_role == MOE_MEASURE_OUTPUT_ROLE_PREFIX);
    REQUIRE(summary.header.projector_signature == include_header.projector_signature);
    REQUIRE(summary.header.media_config_hash == include_header.media_config_hash);
    REQUIRE(summary.header.input_scope == MOE_MEASURE_INPUT_SCOPE_MEDIA);
    REQUIRE(summary.header.output_role == MOE_MEASURE_OUTPUT_ROLE_PREFIX);
    REQUIRE(summary.header.vocab_mask_hash == include_header.vocab_mask_hash);

    {
        std::ofstream tail(generated_path, std::ios::binary | std::ios::app);
        tail.write("partial", 7);
    }
    REQUIRE(moe_measure_measurement_read(generated_path, summary, {}, error));
    REQUIRE(summary.has_partial_tail);
    REQUIRE(moe_measure_measurement_truncate_tail(generated_path, summary, error));
    REQUIRE(moe_measure_measurement_read(generated_path, summary, {}, error));
    REQUIRE(!summary.has_partial_tail && summary.n_blocks == 1);
    const std::string corrupt_path = (root / "corrupt.moem").string();
    std::filesystem::copy_file(generated_path, corrupt_path);
    {
        std::fstream corrupt(corrupt_path, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(-1, std::ios::end);
        const char changed = 0x55;
        corrupt.write(&changed, 1);
    }
    REQUIRE(!moe_measure_measurement_read(corrupt_path, summary, {}, error));

    const moe_measure_measurement_header exclude_header = header({ 3, 4 });
    REQUIRE(moe_measure_measurement_create(exclude_path, exclude_header, error));
    REQUIRE(moe_measure_measurement_append(exclude_path, exclude_header, block(20, 0, 1.0f, 2, 1.0f), error));

    moe_measure_measurement_header old_header = header({ 5, 6 });
    old_header.format_version          = 4;
    REQUIRE(!moe_measure_measurement_create((root / "old.moem").string(), old_header, error));
    const std::string old_file_path = (root / "old-file.moem").string();
    std::filesystem::copy_file(generated_path, old_file_path);
    {
        std::fstream old_file(old_file_path, std::ios::binary | std::ios::in | std::ios::out);
        const uint32_t old_version = 4;
        old_file.seekp(8);
        old_file.write(reinterpret_cast<const char *>(&old_version), sizeof(old_version));
    }
    REQUIRE(!moe_measure_measurement_read(old_file_path, summary, {}, error));


    std::filesystem::remove_all(root);
    std::cout << "MoE measurement format tests passed\n";
    return 0;
}
