// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <stdexcept>
#include <utility>
#include <ff/distributed/dsender/batch.hpp>
#include <ff/distributed/dsender/send_pool.hpp>
#include <ff/distributed/protocol/tokens.hpp>

namespace ff::distributed::sender {

/** One remote connection with its batch, send operations and token window.
 * MTCL owns transport ordering. This class owns the bytes referenced by pending
 * sends and keeps their handle alive. Sending requires an available token;
 * the Sender coordinates progress across sessions while waiting for one.
 * Token confirmations and local send completion are deliberately separate.
 */
class Session {
    // Declaration order keeps the handle alive until all operations are destroyed.
    MTCL::HandleUser handle_;
    SendTokens tokens_;
    SendPool sends_;
    Batch batch_;
public:
    Session(MTCL::HandleUser handle, const SenderOptions& options)
        : handle_(std::move(handle)), tokens_(options.protocol.flowControl, options.sendTokens),
          sends_(options.sendBuffers) {}
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;
    ~Session() = default;

    bool hasToken() const { return tokens_.available(); }
    std::size_t outstanding() const { return tokens_.outstanding(); }
    Batch& batch() { return batch_; }
    const Batch& batch() const { return batch_; }
    bool progressSends() { return sends_.progress(); }
    void completeSends() { sends_.wait(); }
    void closeWrites() { handle_.close(); }

    void pollConfirmation() {
        if (!tokens_.enabled()) return;
        std::size_t size = 0;
        const auto result = handle_.probe(size, false);
        if (result < 0 && wouldBlock()) return;
        if (result < 0) transportError("distributed confirmation probe");
        if (!result) {
            if (tokens_.outstanding())
                throw std::runtime_error("session closed with confirmations outstanding");
            return;
        }
        if (size != sizeof(ack_t)) throw std::runtime_error("invalid distributed confirmation size");
        ack_t confirmation;
        if (handle_.receive(&confirmation, sizeof(confirmation)) != sizeof(confirmation))
            transportError("distributed confirmation receive");
        if (confirmation.ack != ack_t{}.ack)
            throw std::runtime_error("invalid distributed confirmation");
        tokens_.confirm();
    }

    // The caller first obtains a token, progressing all sessions while waiting.
    // encode() moves the batch into the operation's storage and empties the batch.
    void sendBatch() {
        if (batch_.empty()) return;
        auto& operation = sends_.acquire();
        batch_.encode(operation.storage());
        tokens_.acquire();
        operation.start(handle_);
    }

    void sendSingle(const message2_t& message, std::size_t separatePayloadAbove) {
        const bool separate = message.size > separatePayloadAbove;
        auto& operation = sends_.acquire();
        Batch::encodeSingle(message, separate, operation.storage());
        tokens_.acquire();
        operation.start(handle_);
        // Preserve the existing two-frame wire protocol. Blocking send keeps the
        // borrowed payload alive and follows the descriptor in MTCL's send queue.
        if (separate && handle_.send(message.data, message.size) != static_cast<ssize_t>(message.size))
            transportError("distributed large payload send");
    }
};

} // namespace ff::distributed::sender
