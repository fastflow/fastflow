// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <optional>
#include <stdexcept>
#include <utility>
#include <ff/distributed/dreceiver/decoder.hpp>
#include <ff/distributed/dreceiver/delivery.hpp>
#include <ff/distributed/protocol/mtcl.hpp>
#include <ff/distributed/protocol/options.hpp>

namespace ff::distributed::receiver {

// Await states may probe for a frame. Receiving states may only progress their
// existing request. AwaitPayload admits exactly the descriptor's next payload.
enum class ReceivePhase { AwaitBatch, ReceivingBatch, AwaitPayload, ReceivingPayload, Closed };
enum class BatchAdmission { AcceptNew, DrainPending };

/** Receive-side state for one MTCL connection.
 * At most one receive is posted per session. probe() is never called while that
 * receive is pending, because MTCL may still expose its cached frame size.
 * A large-record descriptor reserves the following frame for its payload;
 * other sessions can continue independently while this session waits.
 * Buffer storage and the request remain alive together until completion.
 */
class Session {
    MTCL::HandleUser handle_;
    ProtocolOptions options_;
    ReceiveBuffer buffer_;
    MTCL::Request request_;
    ReceivePhase phase_ = ReceivePhase::AwaitBatch;
    std::size_t expectedBytes_ = 0;
    std::optional<wire::RecordHeader> separatePayload_;

    bool receiving() const {
        return phase_ == ReceivePhase::ReceivingBatch || phase_ == ReceivePhase::ReceivingPayload;
    }

    void acknowledge() {
        if (options_.flowControl != FlowControl::Tokens) return;
        const ack_t confirmation{};
        if (handle_.send(&confirmation, sizeof(confirmation)) != sizeof(confirmation))
            transportError("distributed confirmation send");
    }

    void startReceive() {
        std::size_t bytes = 0;
        const auto result = handle_.probe(bytes, false);
        if (result < 0 && wouldBlock()) return;
        if (result < 0) transportError("distributed batch probe");
        if (!result) {
            if (phase_ == ReceivePhase::AwaitPayload)
                throw std::runtime_error("session closed before its separate payload");
            phase_ = ReceivePhase::Closed;
            handle_.close();
            return;
        }
        if (phase_ == ReceivePhase::AwaitPayload && bytes != separatePayload_->payloadBytes)
            throw std::runtime_error("separate payload has an unexpected length");
        buffer_.ensureCapacity(bytes);
        expectedBytes_ = bytes;
        const auto next = phase_ == ReceivePhase::AwaitBatch
            ? ReceivePhase::ReceivingBatch : ReceivePhase::ReceivingPayload;
        phase_ = next;
        if (handle_.ireceive(buffer_.data(), buffer_.capacity(), request_) < 0)
            transportError("distributed ireceive");
    }

    void verifyReceive() {
        complete(request_);
        const auto received = request_.count();
        if (received < 0 || static_cast<std::size_t>(received) != expectedBytes_)
            throw std::runtime_error("distributed receive length mismatch");
    }

    template<class Deliver> void deliverSeparatePayload(Deliver& deliver) {
        const auto record = *separatePayload_;
        separatePayload_.reset();
        phase_ = ReceivePhase::AwaitBatch;
        deliver(makeReceivedMessage(record, buffer_.release()));
    }

    template<class Deliver> void processBatch(Deliver& deliver) {
        const auto decoded = decodeBatch(buffer_.data(), expectedBytes_, options_.separatePayloadAbove);
        // Preserve the existing confirmation timing during the structural change.
        // Moving this after local delivery needs feedback/deadlock validation.
        acknowledge();
        if (decoded.separatePayload) {
            separatePayload_ = decoded.separatePayload;
            phase_ = ReceivePhase::AwaitPayload;
            return;
        }

        phase_ = ReceivePhase::AwaitBatch;
        if (decoded.records.size() == 1 && decoded.records.front().payloadBytes) {
            deliver(makeReceivedMessage(decoded.records.front(), buffer_.release()));
            return;
        }
        const char* payload = buffer_.data();
        for (const auto& record : decoded.records) {
            deliver(copyReceivedMessage(record, payload));
            payload += record.payloadBytes;
        }
    }

public:
    Session(MTCL::HandleUser handle, ProtocolOptions options)
        : handle_(std::move(handle)), options_(options) {
        if (!options.separatePayloadAbove)
            throw std::invalid_argument("separate payload threshold must be positive");
    }
    Session(const Session&) = delete;
    Session& operator=(const Session&) = delete;
    Session(Session&&) = delete;
    Session& operator=(Session&&) = delete;
    ~Session() {
        if (receiving()) (void)MTCL::wait(request_);
    }

    bool hasPendingBatch() const {
        return phase_ != ReceivePhase::AwaitBatch && phase_ != ReceivePhase::Closed;
    }

    // deliver takes ownership of each message. DrainPending completes only work
    // already started; it still admits the payload belonging to a descriptor.
    template<class Deliver> void progress(BatchAdmission admission, Deliver&& deliver) {
        if (phase_ == ReceivePhase::Closed) return;
        if (!receiving()) {
            if (phase_ == ReceivePhase::AwaitBatch && admission == BatchAdmission::DrainPending) return;
            startReceive();
        }
        if (!receiving() || !MTCL::test(request_)) return;
        verifyReceive();
        if (phase_ == ReceivePhase::ReceivingPayload) deliverSeparatePayload(deliver);
        else processBatch(deliver);
    }
};

} // namespace ff::distributed::receiver
