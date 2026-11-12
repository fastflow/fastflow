// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <ff/distributed/protocol/wire.hpp>

namespace ff::distributed::sender {

/** Accumulates copied records without owning transport state.
 * encode() writes the existing payload/header/count layout into caller-owned
 * storage. clear() preserves capacity for subsequent batches. Large payloads
 * bypass accumulation and use encodeSingle() for their descriptor frame.
 */
class Batch {
    std::vector<char> payloads_;
    std::vector<wire::RecordHeader> records_;
public:
    std::size_t recordCount() const { return records_.size(); }
    std::size_t payloadBytes() const { return payloads_.size(); }
    bool empty() const { return records_.empty(); }

    void append(const message2_t& message) {
        const auto header = wire::describe(message);
        if (message.size) {
            const auto offset = payloads_.size();
            if (message.size > payloads_.max_size() - offset)
                throw std::length_error("distributed batch payload overflow");
            payloads_.resize(offset + message.size);
            std::memcpy(payloads_.data() + offset, message.data, message.size);
        }
        records_.push_back(header);
    }

    void encode(std::vector<char>& output) const {
        if (records_.size() > (output.max_size() - wire::batchCountBytes) / wire::recordHeaderBytes)
            throw std::length_error("distributed batch header overflow");
        const auto overhead = records_.size() * wire::recordHeaderBytes + wire::batchCountBytes;
        if (payloads_.size() > output.max_size() - overhead)
            throw std::length_error("distributed batch size overflow");
        output.resize(payloads_.size() + overhead);
        if (!payloads_.empty()) std::memcpy(output.data(), payloads_.data(), payloads_.size());
        auto* header = output.data() + payloads_.size();
        for (const auto& record : records_) {
            wire::encodeHeader(header, record);
            header += wire::recordHeaderBytes;
        }
        wire::encodeCount(header, records_.size());
    }

    static void encodeSingle(const message2_t& message, bool separatePayload,
                             std::vector<char>& output) {
        const auto header = wire::describe(message);
        const auto payloadBytes = separatePayload ? 0 : message.size;
        constexpr auto overhead = wire::recordHeaderBytes + wire::batchCountBytes;
        if (payloadBytes > output.max_size() - overhead)
            throw std::length_error("distributed message size overflow");
        output.resize(payloadBytes + overhead);
        if (payloadBytes) std::memcpy(output.data(), message.data, payloadBytes);
        wire::encodeHeader(output.data() + payloadBytes, header);
        wire::encodeCount(output.data() + payloadBytes + wire::recordHeaderBytes, 1);
    }

    void clear() {
        payloads_.clear();
        records_.clear();
    }
};

} // namespace ff::distributed::sender
