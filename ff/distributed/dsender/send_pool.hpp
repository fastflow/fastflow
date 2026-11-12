// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <thread>
#include <vector>
#include <ff/distributed/dsender/send_operation.hpp>

namespace ff::distributed::sender {

/** Bounded, session-owned storage for asynchronous sends.
 * Buffers are reused only after successful completion. acquire() makes progress
 * while the pool is exhausted; it does not allocate unbounded pending storage.
 * All methods are called by the sender thread, so no global pool or lock is needed.
 */
class SendPool {
    std::vector<std::unique_ptr<SendOperation>> operations_;
public:
    explicit SendPool(std::size_t count) {
        if (!count) throw std::invalid_argument("send pool must not be empty");
        operations_.reserve(count);
        for (std::size_t i = 0; i < count; ++i)
            operations_.push_back(std::make_unique<SendOperation>());
    }

    SendOperation& acquire() {
        for (;;) {
            for (const auto& operation : operations_)
                if (operation->poll()) return *operation;
            std::this_thread::yield();
        }
    }

    bool progress() {
        bool finished = true;
        for (const auto& operation : operations_) {
            const bool completed = operation->poll();
            finished = completed && finished;
        }
        return finished;
    }

    void wait() {
        for (const auto& operation : operations_) operation->wait();
    }
};

} // namespace ff::distributed::sender
