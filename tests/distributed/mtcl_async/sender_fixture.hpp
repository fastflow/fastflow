// Tests ff_dsenderMTCL2 with MTCL's TCP handle. Only the connection setup of
// svc_init() is replaced: the sender gets one handle on a local socket pair,
// and the test reads the other end.
//
// fillSendQueue() fills the socket send buffer, so MTCL cannot write and keeps
// the next isend pending. drainSendQueue() reads that filler back without
// calling MTCL, so the isend stays pending. The tests compare the bytes read
// with frames encoded in test_support.hpp.
#ifndef FF_TESTS_MTCL_ASYNC_SENDER_FIXTURE_HPP
#define FF_TESTS_MTCL_ASYNC_SENDER_FIXTURE_HPP

#include "test_support.hpp"

#include <cstdlib>
#include <functional>
#include <memory>
#include <new>
#include <tuple>
#include <unordered_map>

namespace mtcl_async_test {

inline constexpr std::size_t kBatchByteLimit = 128;
inline constexpr std::size_t kSmallPayloadBytes = 32;
inline constexpr std::size_t kMessagePoolSize = 16;

// MTCL connection holding test-created handles; progress, shutdown and close are ConnTcp's own.
class LocalSenderConnection : public MTCL::ConnTcp {
public:
    void attach(MTCL::HandleTCP& handle) { connections.emplace(handle.fd, &handle); }
};

// MTCL's TCP handle. It also counts isend calls and records whether the first stayed pending.
class RecordingTcpHandle : public MTCL::HandleTCP {
public:
    using MTCL::HandleTCP::HandleTCP;
    using MTCL::HandleTCP::isend;

    [[nodiscard]] std::size_t sendCallCount() const noexcept { return sendCallCount_; }
    [[nodiscard]] bool firstSendWasPending() const noexcept { return firstSendWasPending_; }

    // Runs once, after the first isend.
    void afterFirstSend(std::function<void()> action) { afterFirstSend_ = std::move(action); }

    ssize_t isend(const void* buffer, std::size_t size, MTCL::Request& request) override {
        const auto result = MTCL::HandleTCP::isend(buffer, size, request);
        if (result < 0) throwSystemError("MTCL isend");
        if (++sendCallCount_ == 1) {
            firstSendWasPending_ = has_active_send_req();
            if (auto action = std::exchange(afterFirstSend_, {})) action();
        }
        return result;
    }

private:
    std::size_t sendCallCount_ = 0;
    bool firstSendWasPending_ = false;
    std::function<void()> afterFirstSend_;
};

enum class ReceiveState { WouldBlock, PeerClosed };

class TestTransport {
public:
    TestTransport() {
        auto sockets = makeSocketPair();
        auto handle = std::make_unique<RecordingTcpHandle>(&connection_, sockets.mtclEnd.get());
        connection_.attach(*handle);
        // From here the connection owns the MTCL end. Nothing below throws.
        (void)sockets.mtclEnd.release();
        testEnd_ = std::move(sockets.testEnd);
        handle_.reset(handle.release());
    }
    TestTransport(const TestTransport&) = delete;
    TestTransport& operator=(const TestTransport&) = delete;
    TestTransport(TestTransport&&) = delete;            // the handle stores &connection_
    TestTransport& operator=(TestTransport&&) = delete;
    ~TestTransport() = default;

    [[nodiscard]] RecordingTcpHandle& handle() noexcept { return *handle_; }

    // Sends filler until send() returns EAGAIN, so the next isend stays pending.
    void fillSendQueue() {
        require(fillerBytes_ == 0, "socket queue already contains filler");
        std::array<char, 1024> filler{};
        filler.fill(kFiller);
        for (;;) {
            const auto count = ::send(handle_->fd, filler.data(), filler.size(), MSG_NOSIGNAL);
            if (count > 0) {
                fillerBytes_ += static_cast<std::size_t>(count);
                continue;
            }
            if (count == -1 && errno == EINTR) continue;
            if (count == -1 && wouldBlock(errno)) break;
            throwSystemError("send queue filler");
        }
        require(fillerBytes_ > 0, "socket queue accepted no filler");
    }

    // Reads the filler back without calling MTCL, so pending sends stay pending.
    void drainSendQueue() {
        std::array<char, 1024> filler{};
        while (fillerBytes_ > 0) {
            const auto count = receiveSome(testEnd_.get(), filler.data(), std::min(fillerBytes_, filler.size()));
            if (count == -1) throwSystemError("receive queue filler");
            require(count > 0, "peer closed before all filler was drained");
            require(std::all_of(filler.begin(), filler.begin() + count, [](char byte) { return byte == kFiller; }),
                    "received bytes do not match the expected filler prefix");
            fillerBytes_ -= static_cast<std::size_t>(count);
        }
    }

    // Appends every readable byte to wire.
    ReceiveState collectWire(Bytes& wire) {
        require(fillerBytes_ == 0, "drain filler before collecting message bytes");
        std::array<char, 4096> data{};
        for (;;) {
            const auto count = receiveSome(testEnd_.get(), data.data(), data.size());
            if (count > 0) {
                wire.insert(wire.end(), data.begin(), data.begin() + count);
                continue;
            }
            if (count == 0) return ReceiveState::PeerClosed;
            if (wouldBlock(errno)) return ReceiveState::WouldBlock;
            throwSystemError("receive wire bytes");
        }
    }

private:
    struct CloseHandle {
        // The sender's HandleUser is destroyed first; MTCL then closes the
        // descriptor and deletes the handle when its reference count reaches zero.
        void operator()(RecordingTcpHandle* handle) const noexcept { handle->close(true, true); }
    };

    static constexpr char kFiller = '~';
    LocalSenderConnection connection_;
    FileDescriptor testEnd_;
    std::unique_ptr<RecordingTcpHandle, CloseHandle> handle_;
    std::size_t fillerBytes_ = 0;
};

// ff_dsenderMTCL2 with the state svc_init() would build, but for one local
// connection instead of remote endpoints. Token flow control is not compiled in.
class SenderWithOneConnection : public ff::ff_dsenderMTCL2 {
public:
    SenderWithOneConnection(RecordingTcpHandle& handle, int batchCount, std::size_t batchBytes)
        : ff_dsenderMTCL2(kNoEndpoints, kNoChannels, batchCount, batchBytes) {
        ff::uBufferAllocator::init(SEND_BUFFERS_PREALLOCATE,
                                  2 * SINGLE_SEND_SIZE_THRESHOLD + batchCount * ff::headerSize + sizeof(int),
                                  batchCount);
        MTCL_Handlers.emplace_back(&handle, false, false);
        handlerCounters.push_back(DEFAULT_MESSAGE_OTF);
        batchBuffers.emplace_back(batchCount, batchBytes, this, MTCL_Handlers[0], handlerCounters[0]);
        dest2ConnID[kDestinationId] = 0;
        src2PossibleDestConnID[kSourceId].insert(0);
        totalAcks = DEFAULT_MESSAGE_OTF; // all tokens available: svc_end() waits for no ack
    }

private:
    using Endpoints = std::vector<std::tuple<std::string, std::string, std::vector<ff::ff_node*>>>;
    using Channels = std::unordered_map<ff::ff_node*, ff::IngressEgressChannels_t>;
    // The base class keeps references to these.
    inline static const Endpoints kNoEndpoints;
    inline static const Channels kNoChannels;
};

struct ReleaseMessage {
    void operator()(ff::message2_t* message) const noexcept { ff::MessageAllocator::releaseMessage(message); }
};
using Message = std::unique_ptr<ff::message2_t, ReleaseMessage>;

// Scoped FastFlow message allocator. Construct it before any sender or message.
class MessagePool {
public:
    MessagePool() {
        ff::MessageAllocator::init(kMessagePoolSize);
        try {
            initializeChannels();
        } catch (...) {
            ff::MessageAllocator::finalize();
            throw;
        }
    }
    ~MessagePool() { ff::MessageAllocator::finalize(); }
    MessagePool(const MessagePool&) = delete;
    MessagePool& operator=(const MessagePool&) = delete;
    MessagePool(MessagePool&&) = delete;
    MessagePool& operator=(MessagePool&&) = delete;

private:
    // svc() allocates broadcast EOS messages without setting their channel:
    // take every pooled message once and return it initialized.
    static void initializeChannels() {
        std::array<Message, kMessagePoolSize> messages;
        for (auto& message : messages) {
            message.reset(ff::MessageAllocator::allocateMessage());
            message->type = ff::ChannelType::FWD;
            message->locality = ff::ChannelLocality::REMOTE;
        }
    }
};

inline Message makeDataMessage(std::string_view payload) {
    Message message{ff::MessageAllocator::allocateMessage()};
    message->src = kSourceId;
    message->dest = kDestinationId;
    message->type = ff::ChannelType::FWD;
    message->locality = ff::ChannelLocality::REMOTE;
    if (!payload.empty()) {
        // message2_t releases owned payloads with free().
        message->data = static_cast<char*>(std::malloc(payload.size()));
        if (!message->data) throw std::bad_alloc{};
        message->cleanup = true;
        message->size = payload.size();
        std::memcpy(message->data, payload.data(), payload.size());
    }
    return message;
}

inline Message makeLogicalEos() {
    Message message{ff::MessageAllocator::make_logical_EOS(kSourceId)};
    message->type = ff::ChannelType::FWD;
    message->locality = ff::ChannelLocality::REMOTE;
    return message;
}

// Calls MTCL until no send is pending, appending readable bytes to wire.
inline void completePendingSends(TestTransport& transport, Bytes& wire) {
    constexpr std::size_t maxProgressIterations = 10000; // run.sh's timeout bounds blocking calls
    for (std::size_t iteration = 0; iteration < maxProgressIterations; ++iteration) {
        transport.collectWire(wire);
        // Test every request: each test() advances MTCL, so stopping at the first
        // pending one would change how far the sends progress.
        bool buffersComplete = true;
        for (auto* buffer : ff::uBufferAllocator::q) buffersComplete = MTCL::test(buffer->r) && buffersComplete;
        if (buffersComplete && !transport.handle().has_active_send_req()) {
            transport.collectWire(wire);
            return;
        }
    }
    throw HarnessError("requests did not complete within the bounded MTCL progress loop");
}

// The frame the sender writes for one unbatched message.
inline Bytes expectedMessageFrame(std::string_view payload, ff::addr_t destination = kDestinationId) {
    return mtclFrame(singleMessageBatch(payload, destination));
}

} // namespace mtcl_async_test
#endif
