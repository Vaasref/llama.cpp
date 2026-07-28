#include "moe-measure-media.h"

#include <filesystem>
#include <fstream>
#include <iostream>

using json = nlohmann::ordered_json;

namespace {

void require(bool condition, const char * expression, int line) {
    if (!condition) {
        std::cerr << "test failure at line " << line << ": " << expression << '\n';
        exit(1);
    }
}

#define REQUIRE(expression) require((expression), #expression, __LINE__)

json messages_with(const std::string & url) {
    return json::array({
        { { "role", "user" },
         { "content", json::array({ { { "type", "text" }, { "text", "look" } },
                                     { { "type", "image_url" }, { "image_url", { { "url", url } } } } }) } }
    });
}

}  // namespace

int main() {
    const std::filesystem::path root =
        std::filesystem::temp_directory_path() / ("llama-moe-measure-media-test-" + moe_measure_id_string(moe_measure_make_id("test")));
    const std::filesystem::path outside = root.parent_path() / (root.filename().string() + "-outside.bin");
    std::filesystem::create_directories(root / "nested");
    {
        std::ofstream output(root / "nested" / "image.bin", std::ios::binary);
        output.write("image", 5);
    }
    {
        std::ofstream output(outside, std::ios::binary);
        output.write("outside", 7);
    }

    std::vector<moe_measure_media_blob> images;
    std::string                  error;
    json                         local = messages_with("file://nested/image.bin");
    bool                         has_images = false;
    REQUIRE(moe_measure_media_messages_have_images(local, has_images, error));
    REQUIRE(has_images);
    const json text_only = json::array({
        { { "role", "user" }, { "content", "look" } }
    });
    REQUIRE(moe_measure_media_messages_have_images(text_only, has_images, error));
    REQUIRE(!has_images);
    REQUIRE(moe_measure_media_extract_images(local, "<__media__>", root.string(), 64, images, error));
    REQUIRE(images.size() == 1 && images[0].bytes.size() == 5);
    REQUIRE(local[0]["content"][1]["type"] == "media_marker");
    const uint64_t local_digest = images[0].digest;

    json data = messages_with("data:image/png;base64,aW1hZ2U=");
    REQUIRE(moe_measure_media_extract_images(data, "<__media__>", "", 64, images, error));
    REQUIRE(images.size() == 1 && images[0].bytes.size() == 5);
    REQUIRE(images[0].digest == local_digest);

    json multiple = messages_with("data:image/png;base64,aW1hZ2U=");
    multiple[0]["content"].push_back({
        { "type",      "image_url"                              },
        { "image_url", { { "url", "file://nested/image.bin" } } }
    });
    REQUIRE(moe_measure_media_extract_images(multiple, "<__media__>", root.string(), 64, images, error));
    REQUIRE(images.size() == 2);

    json too_large = messages_with("data:image/png;base64,aW1hZ2U=");
    REQUIRE(!moe_measure_media_extract_images(too_large, "<__media__>", "", 4, images, error));
    json remote = messages_with("https://example.invalid/image.png");
    REQUIRE(!moe_measure_media_messages_have_images(remote, has_images, error));
    REQUIRE(!moe_measure_media_extract_images(remote, "<__media__>", root.string(), 64, images, error));
    json raw = messages_with("aW1hZ2U=");
    REQUIRE(!moe_measure_media_extract_images(raw, "<__media__>", root.string(), 64, images, error));
    json malformed = messages_with("data:image/png;base64,%%%%");
    REQUIRE(!moe_measure_media_extract_images(malformed, "<__media__>", root.string(), 64, images, error));
    json audio = json::array({
        { { "role", "user" }, { "content", json::array({ { { "type", "input_audio" } } }) } }
    });
    REQUIRE(!moe_measure_media_messages_have_images(audio, has_images, error));
    REQUIRE(!moe_measure_media_extract_images(audio, "<__media__>", root.string(), 64, images, error));
    json traversal = messages_with("file://../" + outside.filename().string());
    REQUIRE(!moe_measure_media_extract_images(traversal, "<__media__>", root.string(), 64, images, error));

    std::error_code ec;
    std::filesystem::create_symlink(outside, root / "nested" / "escape.bin", ec);
    if (!ec) {
        json symlink = messages_with("file://nested/escape.bin");
        REQUIRE(!moe_measure_media_extract_images(symlink, "<__media__>", root.string(), 64, images, error));
    }

    moe_measure_measurement_header header;
    header.model_signature        = 1;
    header.projector_signature    = 2;
    header.template_hash          = 3;
    header.tokenization_hash      = 4;
    header.media_config_hash      = 5;
    header.media_pipeline_version = 1;
    header.input_scope            = MOE_MEASURE_INPUT_SCOPE_MEDIA;
    json stable                   = messages_with("data:image/png;base64,aW1hZ2U=");
    REQUIRE(moe_measure_media_extract_images(stable, "<__media__>", "", 64, images, error));
    const uint64_t hash = moe_measure_multimodal_context_hash(header, "prompt", images);
    REQUIRE(hash == moe_measure_multimodal_context_hash(header, "prompt", images));
    header.input_scope = MOE_MEASURE_INPUT_SCOPE_ALL;
    REQUIRE(hash != moe_measure_multimodal_context_hash(header, "prompt", images));
    const uint64_t all_hash = moe_measure_multimodal_context_hash(header, "prompt", images);
    header.output_role      = MOE_MEASURE_OUTPUT_ROLE_PREFIX;
    REQUIRE(all_hash != moe_measure_multimodal_context_hash(header, "prompt", images));
    header.output_role     = MOE_MEASURE_OUTPUT_ROLE_PRIMARY;
    header.vocab_mask_hash = 9;
    REQUIRE(all_hash != moe_measure_multimodal_context_hash(header, "prompt", images));
    REQUIRE(moe_measure_prefix_hash_text(0, 7) != moe_measure_prefix_hash_media(0, images[0].digest, 0, 0, 0, 0, 0, 0));
    REQUIRE(moe_measure_prefix_hash_media(0, images[0].digest, 0, 0, 0, 0, 0, 0) !=
            moe_measure_prefix_hash_media(0, images[0].digest, 0, 0, 1, 0, 0, 0));
    REQUIRE(moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_MEDIA, true, true));
    REQUIRE(!moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_MEDIA, true, false));
    REQUIRE(moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_TEXT, true, false));
    REQUIRE(!moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_TEXT, true, true));
    REQUIRE(moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_ALL, true, false));
    REQUIRE(moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_ALL, true, true));
    REQUIRE(moe_measure_scope_selects(MOE_MEASURE_INPUT_SCOPE_MEDIA, false, false));
    REQUIRE(moe_measure_effective_input_scope(false, false, MOE_MEASURE_INPUT_SCOPE_MEDIA) == MOE_MEASURE_INPUT_SCOPE_TEXT);
    REQUIRE(moe_measure_effective_input_scope(false, true, MOE_MEASURE_INPUT_SCOPE_ALL) == MOE_MEASURE_INPUT_SCOPE_TEXT);
    REQUIRE(moe_measure_effective_input_scope(true, false, MOE_MEASURE_INPUT_SCOPE_MEDIA) == MOE_MEASURE_INPUT_SCOPE_MEDIA);
    REQUIRE(moe_measure_effective_input_scope(true, true, MOE_MEASURE_INPUT_SCOPE_MEDIA) == MOE_MEASURE_INPUT_SCOPE_ALL);

    std::filesystem::remove_all(root);
    std::filesystem::remove(outside);
    std::cout << "MoE measurement media parsing and hashing tests passed\n";
    return 0;
}
