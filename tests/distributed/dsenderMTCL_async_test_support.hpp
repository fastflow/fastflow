#ifndef DSENDER_MTCL_ASYNC_TEST_SUPPORT_HPP
#define DSENDER_MTCL_ASYNC_TEST_SUPPORT_HPP

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <iostream>
#include <iterator>
#include <memory>
#include <new>
#include <stdexcept>
#include <string>
#include <system_error>
#include <tuple>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>
#include <arpa/inet.h>
#include <endian.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

// Match ff/dff.hpp's defaults without starting the distributed runtime.
#define DEFAULT_BATCH_SIZE 1
#define DEFAULT_BATCH_BYTE_SIZE 65536
#define DEFAULT_MESSAGE_OTF 100
#define SINGLE_SEND_SIZE_THRESHOLD 524288
#define SEND_BUFFERS_PREALLOCATE 4
#define DFF_ENABLED
#define SINGLE_IO_THREAD
#define MTCL_DISABLE_COLLECTIVES
#include <ff/ff.hpp>
#include <ff/distributed/ff_dsenderMTCL.hpp>

namespace sender_test {
using Bytes = std::vector<char>;
constexpr ff::addr_t kSourceId = 3;
constexpr ff::addr_t kDestinationId = 7;
constexpr std::size_t kBatchByteLimit = 128;
constexpr std::size_t kSmallPayloadBytes = 32;
constexpr std::size_t kMessagePoolSize = 16;
static_assert(sizeof(ff::addr_t) == 2 && sizeof(ff::sizeDFF_t) == 8,
              "Wire fixtures require the 64-bit FastFlow ABI");

struct RegressionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void requireHarness(bool condition, const std::string& detail) {
    if (!condition) throw std::runtime_error(detail);
}

[[noreturn]] inline void throwSystemError(const char* operation) {
    const int error = errno;
    throw std::system_error(error, std::generic_category(), operation);
}

inline bool wouldBlock(int error) noexcept {
    return error == EAGAIN || error == EWOULDBLOCK;
}

class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) noexcept : fd_(fd) {}
    ~FileDescriptor() { close(); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.release()) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            close();
            fd_ = other.release();
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return fd_; }
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

private:
    void close() noexcept {
        if (fd_ != -1) ::close(fd_);
    }
    int fd_;
};

inline std::array<FileDescriptor, 2> makeSocketPair() {
    std::array<int, 2> fds{};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) == -1)
        throwSystemError("socketpair");
    std::array<FileDescriptor, 2> sockets{FileDescriptor{fds[0]}, FileDescriptor{fds[1]}};
    for (const auto& socket : sockets) {
        const int flags = ::fcntl(socket.get(), F_GETFL);
        if (flags == -1) throwSystemError("fcntl(F_GETFL)");
        if (::fcntl(socket.get(), F_SETFL, flags | O_NONBLOCK) == -1)
            throwSystemError("fcntl(F_SETFL)");
    }
    return sockets;
}

// ConnTcp retains its real progress, shutdown and close implementation.
class AttachedTcpConnection : public MTCL::ConnTcp {
public:
    void attach(MTCL::HandleTCP& handle) { connections.emplace(handle.fd, &handle); }
};

class RecordingTcpHandle : public MTCL::HandleTCP {
public:
    using MTCL::HandleTCP::HandleTCP;
    using MTCL::HandleTCP::isend;

    [[nodiscard]] std::size_t sendCallCount() const noexcept { return sendCallCount_; }
    [[nodiscard]] bool firstSendWasPending() const noexcept { return firstSendWasPending_; }
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
        auto handle = std::make_unique<RecordingTcpHandle>(&connection_, sockets[0].get());
        connection_.attach(*handle);
        // From here ConnTcp owns the sending descriptor. Nothing below throws.
        (void)sockets[0].release();
        receiver_ = std::move(sockets[1]);
        handle_.reset(handle.release());
    }

    TestTransport(const TestTransport&) = delete;
    TestTransport& operator=(const TestTransport&) = delete;
    // The handle stores connection_'s address.
    TestTransport(TestTransport&&) = delete;
    TestTransport& operator=(TestTransport&&) = delete;
    ~TestTransport() = default;

    [[nodiscard]] RecordingTcpHandle& handle() noexcept { return *handle_; }

    void fillSendQueue() {
        requireHarness(fillerBytes_ == 0, "socket queue already contains filler");
        std::array<char, 1024> padding{};
        padding.fill(kFiller);
        // Saturate the existing queue; SO_SNDBUF need not have a specific size.
        for (;;) {
            const auto count = ::send(handle_->fd, padding.data(), padding.size(), MSG_NOSIGNAL);
            if (count > 0) {
                fillerBytes_ += static_cast<std::size_t>(count);
                continue;
            }
            if (count == -1 && errno == EINTR) continue;
            if (count == -1 && wouldBlock(errno)) break;
            if (count == -1) throwSystemError("send queue filler");
            throw std::runtime_error("socket accepted zero filler bytes without EAGAIN");
        }
        requireHarness(fillerBytes_ > 0, "socket queue accepted no filler");
    }

    void drainQueueFiller() {
        std::array<char, 1024> padding{};
        while (fillerBytes_ > 0) {
            const auto count = receive(padding.data(), std::min(fillerBytes_, padding.size()));
            if (count == -1) throwSystemError("receive queue filler");
            requireHarness(count > 0, "peer closed before all filler was drained");
            requireHarness(std::all_of(padding.begin(), padding.begin() + count,
                                      [](char byte) { return byte == kFiller; }),
                           "received bytes do not match the expected filler prefix");
            fillerBytes_ -= static_cast<std::size_t>(count);
        }
    }

    ReceiveState collectReceivedBytes(Bytes& wire) {
        requireHarness(fillerBytes_ == 0, "drain filler before collecting message bytes");
        std::array<char, 4096> data{};
        for (;;) {
            const auto count = receive(data.data(), data.size());
            if (count > 0) {
                wire.insert(wire.end(), data.begin(), data.begin() + count);
                continue;
            }
            if (count == 0) return ReceiveState::PeerClosed;
            if (wouldBlock(errno)) return ReceiveState::WouldBlock;
            throwSystemError("receive wire bytes");
        }
    }

    void sendConfirmation() {
        // Independent MTCL wire fixture: native protocol ACK in a sized frame.
        const std::uint64_t size = htobe64(sizeof(ff::ack_t));
        const ff::ack_t confirmation{};
        requireHarness(::send(receiver_.get(), &size, sizeof(size), MSG_NOSIGNAL) == sizeof(size),
                       "failed to send confirmation length");
        requireHarness(::send(receiver_.get(), &confirmation, sizeof(confirmation), MSG_NOSIGNAL) == sizeof(confirmation),
                       "failed to send confirmation payload");
    }

private:
    struct CloseHandle {
        void operator()(RecordingTcpHandle* handle) const noexcept {
            // Sender's HandleUser must be destroyed first. MTCL closes the
            // descriptor and deletes the handle when its reference count is zero.
            handle->close(true, true);
        }
    };

    ssize_t receive(char* data, std::size_t size) {
        ssize_t count;
        do {
            count = ::recv(receiver_.get(), data, size, 0);
        } while (count == -1 && errno == EINTR);
        return count;
    }

    static constexpr char kFiller = '~';
    AttachedTcpConnection connection_;
    FileDescriptor receiver_;
    std::unique_ptr<RecordingTcpHandle, CloseHandle> handle_;
    std::size_t fillerBytes_ = 0;
};

using Endpoints = std::vector<std::tuple<std::string, std::string, std::vector<ff::ff_node*>>>;
using Channels = std::unordered_map<ff::ff_node*, ff::IngressEgressChannels_t>;
inline const Endpoints kEndpoints;
inline const Channels kChannels;

// Supply svc_init()'s state using a local connection instead of remote endpoints.
class SenderWithOneConnection : public ff::ff_dsenderMTCL2 {
public:
    SenderWithOneConnection(RecordingTcpHandle& handle, int batchCount, std::size_t batchBytes)
        : ff_dsenderMTCL2(kEndpoints, kChannels, batchCount, batchBytes) {
        const auto session = attachSession(MTCL::HandleUser(&handle, false, false));
        addDestination(kDestinationId, session);
        allowSource(kSourceId, session);
    }
    bool progress() { return progressSends(); }
};

struct ReleaseMessage {
    void operator()(ff::message2_t* message) const noexcept {
        ff::MessageAllocator::releaseMessage(message);
    }
};
using Message = std::unique_ptr<ff::message2_t, ReleaseMessage>;

class MessagePool {
public:
    MessagePool() { ff::MessageAllocator::init(kMessagePoolSize); }
    ~MessagePool() { ff::MessageAllocator::finalize(); }
    MessagePool(const MessagePool&) = delete;
    MessagePool& operator=(const MessagePool&) = delete;
    MessagePool(MessagePool&&) = delete;
    MessagePool& operator=(MessagePool&&) = delete;
};

inline Message makeDataMessage(const std::string& text) {
    Message message{ff::MessageAllocator::allocateMessage()};
    message->src = kSourceId;
    message->dest = kDestinationId;
    message->type = ff::ChannelType::FWD;
    message->locality = ff::ChannelLocality::REMOTE;
    if (!text.empty()) {
        // message2_t releases owned payloads with free().
        message->data = static_cast<char*>(std::malloc(text.size()));
        if (!message->data) throw std::bad_alloc{};
        message->cleanup = true;
        message->size = text.size();
        std::memcpy(message->data, text.data(), text.size());
    }
    return message;
}

inline Message makeLogicalEos() {
    Message message{ff::MessageAllocator::make_logical_EOS(kSourceId)};
    message->type = ff::ChannelType::FWD;
    message->locality = ff::ChannelLocality::REMOTE;
    return message;
}

template<class T> void append(Bytes& bytes, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* raw = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

// Independent wire fixture: [MTCL length][payload][dest][src][channel][size][count].
inline void appendExpectedMessageFrame(Bytes& wire, const std::string& text,
                                       ff::addr_t destination = kDestinationId) {
    append(wire, htobe64(static_cast<std::uint64_t>(text.size() + ff::headerSize + sizeof(int))));
    wire.insert(wire.end(), text.begin(), text.end());
    append(wire, htons(static_cast<std::uint16_t>(destination)));
    append(wire, htons(static_cast<std::uint16_t>(kSourceId)));
    append(wire, ff::ChannelType::FWD);
    append(wire, htobe64(static_cast<std::uint64_t>(text.size())));
    append(wire, htonl(std::uint32_t{1}));
}

inline void completePendingSendsAndCollectWire(TestTransport& transport, SenderWithOneConnection& sender, Bytes& wire) {
    // Bound cooperative progress; the runner's timeout also bounds blocking calls.
    constexpr std::size_t maxProgressIterations = 10000;
    for (std::size_t iteration = 0; iteration < maxProgressIterations; ++iteration) {
        transport.collectReceivedBytes(wire);
        const bool complete = sender.progress();
        if (complete && !transport.handle().has_active_send_req()) {
            transport.collectReceivedBytes(wire);
            return;
        }
    }
    throw std::runtime_error("requests did not complete during bounded real MTCL progress");
}

inline void expectWireEquals(const Bytes& actual, const Bytes& expected, const char* description) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
    if (mismatch.first != actual.end() || mismatch.second != expected.end()) {
        const auto offset = std::distance(actual.begin(), mismatch.first);
        throw RegressionFailure(std::string(description) + ": first mismatch at byte " +
                                std::to_string(offset) + ", expected " + std::to_string(expected.size()) +
                                " bytes, received " + std::to_string(actual.size()));
    }
}
} // namespace sender_test
#endif
