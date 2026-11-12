// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>
#include <ff/distributed/protocol/wire.hpp>

namespace ff::distributed::receiver {

struct DecodedBatch {
    std::vector<wire::RecordHeader> records;
    std::optional<wire::RecordHeader> separatePayload;
};

/** Validate a complete batch frame without transferring or delivering its data.
 * The returned headers are values; they do not borrow frame memory. A separate
 * payload descriptor reserves the next MTCL frame on this session. All lengths
 * and offsets are validated before the caller may deliver the first record.
 */
inline DecodedBatch decodeBatch(const char* frame, std::size_t bytes, std::size_t separatePayloadAbove) {
    if (!frame || bytes < wire::recordHeaderBytes + wire::batchCountBytes)
        throw std::runtime_error("distributed batch is truncated");
    const auto count = wire::decodeCount(frame + bytes - wire::batchCountBytes);
    const auto contentBytes = bytes - wire::batchCountBytes;
    if (!count || count > static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        count > contentBytes / wire::recordHeaderBytes)
        throw std::runtime_error("invalid distributed batch count");
    const auto payloadBytes = contentBytes - count * wire::recordHeaderBytes;
    const auto* headers = frame + payloadBytes;
    DecodedBatch decoded;
    decoded.records.reserve(count);
    std::size_t consumed = 0;
    for (std::size_t index = 0; index < count; ++index) {
        const auto record = wire::decodeHeader(headers + index * wire::recordHeaderBytes);
        if (record.payloadBytes > separatePayloadAbove) {
            if (count != 1 || payloadBytes != 0)
                throw std::runtime_error("invalid separate payload descriptor");
            decoded.separatePayload = record;
        } else {
            if (record.payloadBytes > payloadBytes - consumed)
                throw std::runtime_error("distributed payload exceeds its batch");
            consumed += record.payloadBytes;
        }
        decoded.records.push_back(record);
    }
    if (consumed != payloadBytes) throw std::runtime_error("unexpected distributed batch bytes");
    return decoded;
}

} // namespace ff::distributed::receiver
