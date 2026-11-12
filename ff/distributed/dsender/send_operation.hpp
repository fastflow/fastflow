// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <stdexcept>
#include <vector>
#include <ff/distributed/protocol/mtcl.hpp>

namespace ff::distributed::sender {

/** Owns the storage and request for one asynchronous send.
 * Pending storage cannot be edited, and the object cannot move while MTCL may
 * retain its address. poll() checks success before making the storage reusable.
 * The session must outlive the operation; destruction waits as a final safeguard.
 */
class SendOperation {
    std::vector<char> bytes_;
    MTCL::Request request_;
    enum class State { Available, Pending };
    State state_ = State::Available;
public:
    SendOperation() = default;
    SendOperation(const SendOperation&) = delete;
    SendOperation& operator=(const SendOperation&) = delete;
    SendOperation(SendOperation&&) = delete;
    SendOperation& operator=(SendOperation&&) = delete;
    ~SendOperation() {
        // Owners destroy operations before the associated MTCL handle.
        if (state_ == State::Pending) (void)MTCL::wait(request_);
    }

    bool poll() {
        if (state_ == State::Available) return true;
        if (!MTCL::test(request_)) return false;
        wait();
        return true;
    }

    void wait() {
        if (state_ == State::Available) return;
        // Failure also terminates this request. Do not retry it on destruction.
        const auto result = MTCL::wait(request_);
        state_ = State::Available;
        if (result < 0) transportError("distributed send completion");
    }

    // Borrowed until start(). Keeping this reference across start() is invalid.
    std::vector<char>& storage() {
        if (state_ != State::Available)
            throw std::logic_error("pending send storage is immutable");
        return bytes_;
    }

    void start(MTCL::HandleUser& handle) {
        if (state_ != State::Available)
            throw std::logic_error("send operation is already pending");
        // Even a failed post can leave a request referencing the storage.
        state_ = State::Pending;
        if (handle.isend(bytes_.data(), bytes_.size(), request_) < 0)
            transportError("distributed isend");
    }
};

} // namespace ff::distributed::sender
