// Host LibTorch stream ownership only. No camera/device-control interface.
#include <c10/core/StreamGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <cstdio>
#include <exception>

struct WorkerStream {
    c10::cuda::CUDAStream stream = c10::cuda::getStreamFromPool(false, 0);
    c10::StreamGuard guard{stream.unwrap()};
};

extern "C" void* buttercup_sam_stream_enter(char* error, size_t size) noexcept {
    try {
        return new WorkerStream;
    } catch (const std::exception& e) {
        std::snprintf(error, size, "%s", e.what());
    } catch (...) {
        std::snprintf(error, size, "unknown CUDA stream initialization error");
    }
    return nullptr;
}

extern "C" long long buttercup_sam_stream_id(void* handle) noexcept {
    return static_cast<WorkerStream*>(handle)->stream.id();
}

extern "C" void buttercup_sam_stream_leave(void* handle) noexcept {
    auto* owner = static_cast<WorkerStream*>(handle);
    try {
        // Dispatch through LibTorch's runtime, not a separately linked toolkit
        // libcudart (the system toolkit may differ from LibTorch's CUDA build).
        owner->stream.unwrap().synchronize();
    } catch (const std::exception& e) {
        std::fprintf(stderr, "SAM31 stream shutdown: %s\n", e.what());
    } catch (...) {
        std::fprintf(stderr, "SAM31 stream shutdown: unknown error\n");
    }
    delete owner;
}
