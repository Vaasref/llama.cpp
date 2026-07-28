#include "moe-measure-media.h"

#include "base64.hpp"

#include <filesystem>
#include <fstream>
#include <iterator>

namespace {

bool path_is_within(const std::filesystem::path & root, const std::filesystem::path & candidate) {
    auto root_it      = root.begin();
    auto candidate_it = candidate.begin();
    for (; root_it != root.end(); ++root_it, ++candidate_it) {
        if (candidate_it == candidate.end() || *root_it != *candidate_it) {
            return false;
        }
    }
    return true;
}

bool load_file_url(const std::string &    url,
                   const std::string &    media_path,
                   size_t                 max_bytes,
                   std::vector<uint8_t> & bytes,
                   std::string &          error) {
    if (media_path.empty()) {
        error = "file image_url requires --media-path";
        return false;
    }
    const std::string           relative_text = url.substr(7);
    const std::filesystem::path relative(relative_text);
    if (relative_text.empty() || relative.is_absolute() || relative_text.find('%') != std::string::npos) {
        error = "file image_url must contain an unescaped relative path";
        return false;
    }
    for (const std::filesystem::path & part : relative) {
        if (part == "..") {
            error = "file image_url path traversal is not allowed";
            return false;
        }
    }
    std::error_code             ec;
    const std::filesystem::path root = std::filesystem::canonical(media_path, ec);
    if (ec || !std::filesystem::is_directory(root)) {
        error = "--media-path is not a readable directory";
        return false;
    }
    const std::filesystem::path candidate = std::filesystem::canonical(root / relative, ec);
    if (ec || !path_is_within(root, candidate) || !std::filesystem::is_regular_file(candidate)) {
        error = "file image_url escapes --media-path or is not a regular file";
        return false;
    }
    const uintmax_t size = std::filesystem::file_size(candidate, ec);
    if (ec || size > max_bytes) {
        error = ec ? "failed to determine image size" : "image exceeds --media-max-bytes";
        return false;
    }
    std::ifstream input(candidate, std::ios::binary);
    if (!input) {
        error = "failed to open local image";
        return false;
    }
    bytes.assign(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
    if ((!input.good() && !input.eof()) || bytes.empty()) {
        error = "failed to read local image";
        return false;
    }
    return true;
}

bool load_data_uri(const std::string & url, size_t max_bytes, std::vector<uint8_t> & bytes, std::string & error) {
    const size_t comma = url.find(',');
    if (comma == std::string::npos || comma <= 11 || url.compare(0, 11, "data:image/") != 0 ||
        url.substr(0, comma).size() < 7 || url.substr(0, comma).rfind(";base64") != url.substr(0, comma).size() - 7) {
        error = "image data URI must be data:image/...;base64,...";
        return false;
    }
    const std::string encoded = url.substr(comma + 1);
    if (encoded.empty() || base64::max_decode_size(encoded.size()) > max_bytes) {
        error = encoded.empty() ? "image data URI is empty" : "image exceeds --media-max-bytes";
        return false;
    }
    try {
        bytes.clear();
        bytes.reserve(base64::max_decode_size(encoded.size()));
        base64::decode(encoded.begin(), encoded.end(), std::back_inserter(bytes), base64::alphabet::standard);
    } catch (const std::exception & exception) {
        error = std::string("invalid image data URI: ") + exception.what();
        return false;
    }
    if (bytes.empty() || bytes.size() > max_bytes) {
        error = bytes.empty() ? "image data URI is empty" : "image exceeds --media-max-bytes";
        return false;
    }
    return true;
}

}  // namespace

bool moe_measure_media_messages_have_images(const nlohmann::ordered_json & messages,
                                     bool &                         has_images,
                                     std::string &                  error) {
    has_images = false;
    if (!messages.is_array()) {
        error = "chat record messages must be an array";
        return false;
    }
    for (const auto & message : messages) {
        if (!message.is_object() || !message.contains("content")) {
            continue;
        }
        const auto & content = message["content"];
        if (content.is_string() || content.is_null()) {
            continue;
        }
        if (!content.is_array()) {
            error = "message content must be a string, null, or array";
            return false;
        }
        for (const auto & part : content) {
            if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
                error = "content part must contain a string type";
                return false;
            }
            const std::string type = part["type"];
            if (type == "text") {
                continue;
            }
            if (type == "input_audio" || type == "audio" || type == "input_video" || type == "video_url") {
                error = "audio and video content are not supported";
                return false;
            }
            if (type != "image_url") {
                error = "unsupported content part type: " + type;
                return false;
            }
            if (!part.contains("image_url") || !part["image_url"].is_object() ||
                !part["image_url"].contains("url") || !part["image_url"]["url"].is_string()) {
                error = "image_url content requires image_url.url";
                return false;
            }
            const std::string url = part["image_url"]["url"];
            if (url.compare(0, 11, "data:image/") != 0 && url.compare(0, 7, "file://") != 0) {
                error = url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0 ?
                            "HTTP image URLs are not supported" :
                            "image_url must be a data image URI or file://relative/path";
                return false;
            }
            has_images = true;
        }
    }
    return true;
}

bool moe_measure_media_extract_images(nlohmann::ordered_json &       messages,
                               const std::string &            marker,
                               const std::string &            media_path,
                               size_t                         max_bytes,
                               std::vector<moe_measure_media_blob> & images,
                               std::string &                  error) {
    images.clear();
    if (!messages.is_array()) {
        error = "chat record messages must be an array";
        return false;
    }
    for (auto & message : messages) {
        if (!message.is_object() || !message.contains("content")) {
            continue;
        }
        auto & content = message["content"];
        if (content.is_string() || content.is_null()) {
            continue;
        }
        if (!content.is_array()) {
            error = "message content must be a string, null, or array";
            return false;
        }
        for (auto & part : content) {
            if (!part.is_object() || !part.contains("type") || !part["type"].is_string()) {
                error = "content part must contain a string type";
                return false;
            }
            const std::string type = part["type"];
            if (type == "text") {
                continue;
            }
            if (type == "input_audio" || type == "audio" || type == "input_video" || type == "video_url") {
                error = "audio and video content are not supported";
                return false;
            }
            if (type != "image_url") {
                error = "unsupported content part type: " + type;
                return false;
            }
            if (!part.contains("image_url") || !part["image_url"].is_object() || !part["image_url"].contains("url") ||
                !part["image_url"]["url"].is_string()) {
                error = "image_url content requires image_url.url";
                return false;
            }
            const std::string url = part["image_url"]["url"];
            moe_measure_media_blob   blob;
            if (url.compare(0, 11, "data:image/") == 0) {
                if (!load_data_uri(url, max_bytes, blob.bytes, error)) {
                    return false;
                }
            } else if (url.compare(0, 7, "file://") == 0) {
                if (!load_file_url(url, media_path, max_bytes, blob.bytes, error)) {
                    return false;
                }
            } else if (url.compare(0, 7, "http://") == 0 || url.compare(0, 8, "https://") == 0) {
                error = "HTTP image URLs are not supported";
                return false;
            } else {
                error = "image_url must be a data image URI or file://relative/path";
                return false;
            }
            blob.digest = moe_measure_hash_bytes(0, blob.bytes.data(), blob.bytes.size());
            images.push_back(std::move(blob));
            part = {
                { "type", "media_marker" },
                { "text", marker         }
            };
        }
    }
    return true;
}

uint64_t moe_measure_multimodal_context_hash(const moe_measure_measurement_header &      header,
                                      const std::string &                  prompt,
                                      const std::vector<moe_measure_media_blob> & images) {
    uint64_t hash = moe_measure_hash_bytes(0, &header.model_signature, sizeof(header.model_signature));
    hash          = moe_measure_hash_bytes(hash, &header.projector_signature, sizeof(header.projector_signature));
    hash          = moe_measure_hash_bytes(hash, &header.template_hash, sizeof(header.template_hash));
    hash          = moe_measure_hash_bytes(hash, &header.tokenization_hash, sizeof(header.tokenization_hash));
    hash          = moe_measure_hash_bytes(hash, &header.media_config_hash, sizeof(header.media_config_hash));
    hash          = moe_measure_hash_bytes(hash, &header.media_pipeline_version, sizeof(header.media_pipeline_version));
    hash          = moe_measure_hash_bytes(hash, &header.input_scope, sizeof(header.input_scope));
    hash = moe_measure_hash_bytes(hash, &header.output_role, sizeof(header.output_role));
    hash = moe_measure_hash_bytes(hash, &header.vocab_mask_hash, sizeof(header.vocab_mask_hash));
    hash = moe_measure_hash_string(hash, prompt);
    for (const moe_measure_media_blob & image : images) {
        hash = moe_measure_hash_bytes(hash, &image.digest, sizeof(image.digest));
        hash = moe_measure_hash_bytes(hash, image.bytes.data(), image.bytes.size());
    }
    return hash;
}

uint64_t moe_measure_prefix_hash_text(uint64_t hash, int32_t token) {
    const uint8_t type = 1;
    hash               = moe_measure_hash_bytes(hash, &type, sizeof(type));
    return moe_measure_hash_bytes(hash, &token, sizeof(token));
}

uint64_t moe_measure_prefix_hash_media(uint64_t hash,
                                uint64_t media_digest,
                                uint32_t chunk_index,
                                uint32_t token_index,
                                uint32_t decoder_t,
                                uint32_t decoder_x,
                                uint32_t decoder_y,
                                uint32_t decoder_z) {
    const uint8_t type = 2;
    hash               = moe_measure_hash_bytes(hash, &type, sizeof(type));
    hash               = moe_measure_hash_bytes(hash, &media_digest, sizeof(media_digest));
    hash               = moe_measure_hash_bytes(hash, &chunk_index, sizeof(chunk_index));
    hash               = moe_measure_hash_bytes(hash, &token_index, sizeof(token_index));
    hash               = moe_measure_hash_bytes(hash, &decoder_t, sizeof(decoder_t));
    hash               = moe_measure_hash_bytes(hash, &decoder_x, sizeof(decoder_x));
    hash               = moe_measure_hash_bytes(hash, &decoder_y, sizeof(decoder_y));
    return moe_measure_hash_bytes(hash, &decoder_z, sizeof(decoder_z));
}

bool moe_measure_scope_selects(moe_measure_input_scope scope, bool media_record, bool media_position) {
    if (!media_record) {
        return !media_position;
    }
    return scope == MOE_MEASURE_INPUT_SCOPE_ALL || (scope == MOE_MEASURE_INPUT_SCOPE_MEDIA && media_position) ||
           (scope == MOE_MEASURE_INPUT_SCOPE_TEXT && !media_position);
}

moe_measure_input_scope moe_measure_effective_input_scope(bool has_media,
                                            bool has_paired_media,
                                            moe_measure_input_scope requested_scope) {
    if (!has_media) {
        return MOE_MEASURE_INPUT_SCOPE_TEXT;
    }
    return has_paired_media ? MOE_MEASURE_INPUT_SCOPE_ALL : requested_scope;
}
