// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <cstdlib>
#include <memory>
#include <new>
#include <utility>

namespace ff::distributed::receiver {

struct FreePayload {
    void operator()(char* memory) const noexcept { std::free(memory); }
};
using Payload = std::unique_ptr<char, FreePayload>;

inline Payload allocatePayload(std::size_t bytes) {
    // message2_t's existing ownership ABI requires malloc/free storage.
    Payload payload(static_cast<char*>(std::malloc(bytes)));
    if (!payload && bytes) throw std::bad_alloc{};
    return payload;
}

/** Owns malloc-compatible receive memory and its actual allocation capacity.
 * ensureCapacity() is called only when no receive uses this memory.
 * release() transfers ownership to a delivered message and resets capacity,
 * so the next receive cannot mistake old metadata for a valid allocation.
 */
class ReceiveBuffer {
    Payload memory_;
    std::size_t capacity_ = 0;
public:
    char* data() { return memory_.get(); }
    const char* data() const { return memory_.get(); }
    std::size_t capacity() const { return capacity_; }
    void ensureCapacity(std::size_t bytes) {
        if (bytes <= capacity_) return;
        auto replacement = allocatePayload(bytes);
        memory_ = std::move(replacement);
        capacity_ = bytes;
    }
    Payload release() {
        capacity_ = 0;
        return std::move(memory_);
    }
};

} // namespace ff::distributed::receiver
