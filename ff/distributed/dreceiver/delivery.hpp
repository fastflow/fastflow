// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstring>
#include <utility>
#include <ff/distributed/dreceiver/receive_buffer.hpp>
#include <ff/distributed/protocol/message.hpp>
#include <ff/distributed/protocol/wire.hpp>

namespace ff::distributed::receiver {

// Ownership boundary between transport storage and local FastFlow routing.
// Both data and producer-end records retain their source and destination.
inline Message makeReceivedMessage(const wire::RecordHeader& record, Payload payload) {
    Message message(MessageAllocator::allocateMessage());
    message->src = record.source;
    message->dest = record.destination;
    message->type = record.payloadBytes ? record.channel : ChannelType::FWD;
    message->locality = ChannelLocality::REMOTE;
    message->size = record.payloadBytes;
    message->data = payload.release();
    message->cleanup = true;
    return message;
}

inline Message copyReceivedMessage(const wire::RecordHeader& record, const char* payload) {
    auto storage = allocatePayload(record.payloadBytes);
    if (record.payloadBytes) std::memcpy(storage.get(), payload, record.payloadBytes);
    return makeReceivedMessage(record, std::move(storage));
}

} // namespace ff::distributed::receiver
