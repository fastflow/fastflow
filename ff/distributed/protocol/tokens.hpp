// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <ff/distributed/protocol/options.hpp>

namespace ff::distributed {

/** Per-session accounting for batch confirmations.
 * While enabled, available tokens plus outstanding batches equals the limit.
 * Transport completion does not change this accounting. Confirmations must be
 * received once per batch; unsolicited or duplicate confirmations are errors.
 * An unrestricted policy never consumes tokens and expects no confirmations.
 */
class SendTokens {
    FlowControl policy_;
    std::size_t limit_;
    std::size_t available_;
public:
    SendTokens(FlowControl policy, std::size_t limit)
        : policy_(policy), limit_(limit), available_(limit) {
        if (enabled() && !limit) throw std::invalid_argument("send token limit must be positive");
    }
    bool enabled() const { return policy_ == FlowControl::Tokens; }
    bool available() const { return !enabled() || available_ != 0; }
    std::size_t outstanding() const { return enabled() ? limit_ - available_ : 0; }
    void acquire() {
        if (!available()) throw std::logic_error("no send token available");
        if (enabled()) --available_;
    }
    void confirm() {
        if (!enabled() || available_ == limit_)
            throw std::runtime_error("unexpected distributed batch confirmation");
        ++available_;
    }
};

} // namespace ff::distributed
