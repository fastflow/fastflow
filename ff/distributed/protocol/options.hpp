// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

/** @file
 * Typed configuration shared by the distributed sender and receiver.
 * Legacy build macros are translated only by ff_dsenderMTCL.hpp and
 * ff_dreceiverMTCL.hpp. The protocol threshold and flow policy must agree at
 * both endpoints; the current wire format carries no version or negotiation.
 */

#include <cstddef>
#include <limits>
#include <stdexcept>

namespace ff::distributed {

enum class FlowControl { Unrestricted, Tokens };
enum class LoadBalancing { RoundRobin, Random };

struct ProtocolOptions {
    // Larger records use a descriptor frame followed by a separate payload frame.
    std::size_t separatePayloadAbove = 524288;
    FlowControl flowControl = FlowControl::Unrestricted;
};

struct SenderOptions {
    ProtocolOptions protocol;
    std::size_t batchRecords = 1;
    std::size_t batchBytes = 65536;
    std::size_t sendBuffers = 2;
    // Tokens bound outstanding batches. The send pool separately bounds locally
    // pending operations; neither setting is a byte limit for the remote graph.
    std::size_t sendTokens = 100;
    int connectRetries = 1500;
    LoadBalancing loadBalancing = LoadBalancing::RoundRobin;

    void validate() const {
        if (!batchRecords || batchRecords > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
            !batchBytes || !sendBuffers || connectRetries <= 0 ||
            !protocol.separatePayloadAbove ||
            (protocol.flowControl == FlowControl::Tokens && !sendTokens))
            throw std::invalid_argument("invalid distributed sender options");
    }
};

} // namespace ff::distributed
