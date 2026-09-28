// NInfer C ABI — drive the ninfer Engine from a host process (e.g. ComfyUI's Python).
//
// Why this exists
// ---------------
// ninfer::Engine is a C++ library (see include/ninfer/engine.h). A Python host cannot
// link it directly, and building a CPython extension would pin the DLL to a single
// Python ABI. This target instead exposes a small extern "C" surface that ctypes/cffi
// can load into ANY Python process.
//
// The DLL is loaded *into* the host process, so the engine shares that process's address
// space and CUDA context — no IPC, no subprocess, no second GPU context to synchronise.
//
// Two options structs are used rather than long parameter lists, because they are plain
// POD and ctypes can mirror their layout exactly.
//
// Build: apps/CMakeLists.txt defines the ninfer_capi SHARED target.

#include "ninfer/engine.h"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <string>
#include <utility>
#include <vector>

#if defined(_WIN32)
#  define NINFER_CAPI_API __declspec(dllexport)
#else
#  define NINFER_CAPI_API __attribute__((visibility("default")))
#endif

namespace {

// Accumulates streamed deltas into flat strings. The engine calls publish() while
// generate() is running. The host serialises requests, so no locking is needed here.
struct BufferSink final : ninfer::OutputSink {
    std::string content;
    std::string reasoning;

    void publish(ninfer::OutputDelta delta) override {
        if (delta.channel == ninfer::OutputChannel::Content) {
            content.append(delta.text);
        } else {
            reasoning.append(delta.text);
        }
    }
};

struct Handle {
    ninfer::Engine engine;
    bool vision = false;

    Handle(ninfer::EngineOptions options, bool vision_enabled)
        : engine(std::move(options)), vision(vision_enabled) {}
};

// thread_local so concurrent hosts never clobber each other's diagnostic message.
thread_local std::string g_last_error;

ninfer::ChatMessage text_message(ninfer::ChatRole role, const char* utf8) {
    ninfer::ChatMessage message;
    message.role = role;
    ninfer::MessagePart part;
    part.kind = ninfer::MessagePartKind::Text;
    part.text = (utf8 != nullptr) ? utf8 : "";
    message.parts.push_back(std::move(part));
    return message;
}

} // namespace

extern "C" {

// ---------------------------------------------------------------------------
// Options structs (POD; keep in sync with the ctypes mirrors in the node code).
// ---------------------------------------------------------------------------

typedef struct NinferCreateOptions {
    const char* artifact_path;  // UTF-8 path to the .ninfer artifact; required.
    int32_t device;             // CUDA device index.
    int32_t max_context;        // per-sequence logical context ceiling.
    int32_t speculative_mtp;    // 0 = off; 1..5 = MTP draft tokens.
    int32_t kv_tokens;          // explicit KV capacity; 0 = follow max_context.
    int32_t enable_vision;      // loads the vision encoder (+~2 GB VRAM).
    int32_t vision_max_tokens;  // 0 = runtime default.
    int32_t embedding_host;     // 1 = keep token embedding in pinned host RAM (saves VRAM).
    int32_t kv_dtype;           // 0 = bf16, 1 = int8 group64, 2 = int4 group64.
    int32_t use_cuda_graph;     // 1 = enabled (default).
} NinferCreateOptions;

typedef struct NinferRequestOptions {
    int32_t max_new_tokens;      // <= 0 -> 512.
    int32_t enable_thinking;     // 0 disables the reasoning pass (the fast path).
    float   temperature;         // < 0 -> model default.
    int32_t top_k;               // < 0 -> model default.
    float   top_p;               // < 0 -> model default.
    float   min_p;               // < 0 -> model default.
    float   presence_penalty;    // < 0 -> model default.
    float   frequency_penalty;   // < 0 -> model default.
    int64_t seed;                // < 0 -> random per request.
} NinferRequestOptions;

// ---------------------------------------------------------------------------
// Exported surface
// ---------------------------------------------------------------------------

NINFER_CAPI_API const char* ninfer_last_error(void) {
    return g_last_error.c_str();
}

// Loads an artifact and constructs the engine. This is the slow call (weights -> GPU).
// Returns an opaque handle, or nullptr on failure (see ninfer_last_error()).
NINFER_CAPI_API void* ninfer_create(const NinferCreateOptions* options) {
    g_last_error.clear();
    if (options == nullptr || options->artifact_path == nullptr) {
        g_last_error = "create options (or artifact_path) must not be null";
        return nullptr;
    }

    try {
        const std::uint32_t context =
            static_cast<std::uint32_t>(options->max_context > 0 ? options->max_context : 8192);
        const bool vision = (options->enable_vision != 0);

        ninfer::EngineOptions engine_options;
        // The host passes UTF-8. On Windows std::filesystem::path needs a wide string,
        // and u8string is the portable way to say "these bytes are UTF-8".
        engine_options.artifact_path = std::filesystem::path(
            std::u8string(reinterpret_cast<const char8_t*>(options->artifact_path)));

        engine_options.device      = options->device;
        engine_options.max_context = context;
        engine_options.kv_capacity = ninfer::KvCapacityPolicy::explicit_capacity(
            static_cast<std::uint32_t>(options->kv_tokens > 0 ? options->kv_tokens : context));

        engine_options.enable_vision     = vision;
        engine_options.vision_max_tokens = static_cast<std::uint32_t>(
            options->vision_max_tokens > 0 ? options->vision_max_tokens : 0);
        engine_options.embedding_host    = (options->embedding_host != 0);
        engine_options.use_cuda_graph    = (options->use_cuda_graph != 0);

        switch (options->kv_dtype) {
        case 1:  engine_options.kv_cache = ninfer::KvCacheStorage::Int8Group64; break;
        case 2:  engine_options.kv_cache = ninfer::KvCacheStorage::Int4Group64; break;
        default: engine_options.kv_cache = ninfer::KvCacheStorage::BFloat16;    break;
        }

        if (options->speculative_mtp > 0) {
            engine_options.speculative.backend = ninfer::SpeculativeBackend::Mtp;
            engine_options.speculative.draft_tokens = static_cast<std::uint32_t>(
                options->speculative_mtp > 5 ? 5 : options->speculative_mtp);
        }

        return new Handle(std::move(engine_options), vision);
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return nullptr;
    } catch (...) {
        g_last_error = "unknown error while loading the engine";
        return nullptr;
    }
}

// True when the engine was created with vision enabled.
NINFER_CAPI_API int32_t ninfer_has_vision(void* handle) {
    if (handle == nullptr) {
        return 0;
    }
    return static_cast<Handle*>(handle)->vision ? 1 : 0;
}

// Runs one single-turn generation, with an optional image.
//
// Returns the number of bytes written to out_buf (excluding the terminating NUL), or -1
// on failure. Nothing is thrown across the ABI.
//
//   image_bytes : encoded image payload (JPEG/BMP — the bundled FFmpeg has no PNG
//                 decoder). Pass NULL with image_len 0 for a text-only request; doing so
//                 on an engine created without vision is an error.
//   image_mime  : e.g. "image/jpeg"; NULL defaults to "image/jpeg".
NINFER_CAPI_API int32_t ninfer_generate(void* handle,
                        const char* system_text_utf8,
                        const char* user_text_utf8,
                        const unsigned char* image_bytes,
                        int32_t image_len,
                        const char* image_mime,
                        const NinferRequestOptions* request,
                        char* out_buf,
                        int32_t out_cap) {
    g_last_error.clear();
    if (handle == nullptr) {
        g_last_error = "handle must not be null";
        return -1;
    }
    if (out_buf == nullptr || out_cap <= 0) {
        g_last_error = "output buffer must not be null or empty";
        return -1;
    }

    try {
        auto* self = static_cast<Handle*>(handle);
        const bool has_image = (image_bytes != nullptr && image_len > 0);

        if (has_image && !self->vision) {
            g_last_error =
                "engine was created without vision; recreate it with enable_vision=1";
            return -1;
        }

        ninfer::PromptInput input;
        if (system_text_utf8 != nullptr && *system_text_utf8 != '\0') {
            input.messages.push_back(text_message(ninfer::ChatRole::System, system_text_utf8));
        }

        // User turn: media first, then the instruction text.
        ninfer::ChatMessage user_message;
        user_message.role = ninfer::ChatRole::User;
        if (has_image) {
            ninfer::MessagePart media_part;
            media_part.kind             = ninfer::MessagePartKind::Media;
            media_part.media.kind       = ninfer::MediaKind::Image;
            media_part.media.bytes.assign(image_bytes, image_bytes + image_len);
            media_part.media.media_type = (image_mime != nullptr) ? image_mime : "image/jpeg";
            media_part.media.source_name = "input";
            user_message.parts.push_back(std::move(media_part));
        }
        {
            ninfer::MessagePart text_part;
            text_part.kind = ninfer::MessagePartKind::Text;
            text_part.text = (user_text_utf8 != nullptr) ? user_text_utf8 : "";
            user_message.parts.push_back(std::move(text_part));
        }
        input.messages.push_back(std::move(user_message));

        ninfer::RequestOptions req;
        if (request != nullptr) {
            input.options.enable_thinking = (request->enable_thinking != 0);
            req.execution.requested_output_tokens = static_cast<std::uint32_t>(
                request->max_new_tokens > 0 ? request->max_new_tokens : 512);

            // Negative values mean "leave the registered model/mode default in place".
            if (request->temperature >= 0.0F)       req.execution.sampling.temperature       = request->temperature;
            if (request->top_k >= 0)                req.execution.sampling.top_k             = request->top_k;
            if (request->top_p >= 0.0F)             req.execution.sampling.top_p             = request->top_p;
            if (request->min_p >= 0.0F)             req.execution.sampling.min_p             = request->min_p;
            if (request->presence_penalty >= 0.0F)  req.execution.sampling.presence_penalty  = request->presence_penalty;
            if (request->frequency_penalty >= 0.0F) req.execution.sampling.frequency_penalty = request->frequency_penalty;
            if (request->seed >= 0)                 req.execution.sampling.seed              = static_cast<std::uint64_t>(request->seed);
        } else {
            input.options.enable_thinking         = false;
            req.execution.requested_output_tokens = 512;
        }

        ninfer::PreparedPrompt prompt = self->engine.prepare(std::move(input));

        BufferSink sink;
        self->engine.generate(std::move(prompt), std::move(req), &sink);

        const std::size_t room    = static_cast<std::size_t>(out_cap - 1);
        const std::size_t written = sink.content.size() < room ? sink.content.size() : room;
        if (written > 0) {
            std::memcpy(out_buf, sink.content.data(), written);
        }
        out_buf[written] = '\0';
        return static_cast<int32_t>(written);
    } catch (const std::exception& error) {
        g_last_error = error.what();
        return -1;
    } catch (...) {
        g_last_error = "unknown error during generation";
        return -1;
    }
}

// Destroys the engine and releases every GPU allocation it owns. Call this before handing
// the GPU back to a diffusion pipeline.
NINFER_CAPI_API void ninfer_destroy(void* handle) {
    delete static_cast<Handle*>(handle);
}

} // extern "C"
