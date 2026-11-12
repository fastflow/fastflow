// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <memory>
#include <ff/distributed/ff_messageAllocator.hpp>

namespace ff::distributed {

// Internal meaning of the existing message2_t sentinel representation.
// Empty application payloads cannot be distinguished from producer termination
// in this wire ABI. A flush is local to the sender and is never serialized.
enum class MessageKind { Data, ProducerEnd, Flush };

inline MessageKind messageKind(const message2_t& message) {
    if (message.dest == -2) return MessageKind::Flush;
    if (!message.size) return MessageKind::ProducerEnd;
    return MessageKind::Data;
}

struct ReleaseMessage {
    void operator()(message2_t* message) const {
        if (message) MessageAllocator::releaseMessage(message);
    }
};
// Sole ownership of an allocator-managed envelope and its optional payload owner.
using Message = std::unique_ptr<message2_t, ReleaseMessage>;

inline Message makeProducerEnd(int source, int destination = -1) {
    Message message(MessageAllocator::make_logical_EOS(source, destination));
    message->type = ChannelType::FWD;
    message->locality = ChannelLocality::REMOTE;
    return message;
}

} // namespace ff::distributed
