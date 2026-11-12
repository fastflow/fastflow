#ifndef DRECEIVER_MTCL_ASYNC_TEST_SUPPORT_HPP
#define DRECEIVER_MTCL_ASYNC_TEST_SUPPORT_HPP

#include <cerrno>
#include <cstring>
#include <functional>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>
#include <fcntl.h>
#include <unistd.h>

#define DFF_EXCLUDE_MPI
#define DFF_EXCLUDE_BLOCKING
#include <ff/dff.hpp>

namespace receiver_test {
using Bytes = std::vector<char>;
constexpr ff::addr_t kSource = 3;
constexpr ff::addr_t kDestination = 7;
static_assert(sizeof(ff::addr_t) == 2 && sizeof(ff::sizeDFF_t) == 8,
              "Wire fixtures require the 64-bit FastFlow ABI");

struct RegressionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void expect(bool condition, const std::string& detail) {
    if (!condition) throw RegressionFailure(detail);
}

inline void requireHarness(bool condition, const std::string& detail) {
    if (!condition) throw std::runtime_error(detail);
}

// A batch with one message, constructed independently of the sender:
// [payload][dest:2][src:2][channel:1][payload size:8][message count:4].
template<class T> void append(Bytes& bytes, T value) {
    const char* raw = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

inline Bytes messageFrame(const std::string& payload) {
    Bytes frame(payload.begin(), payload.end());
    append(frame, htons(static_cast<uint16_t>(kDestination)));
    append(frame, htons(static_cast<uint16_t>(kSource)));
    append(frame, ff::ChannelType::FWD);
    append(frame, htobe64(static_cast<uint64_t>(payload.size())));
    append(frame, htonl(uint32_t{1}));
    return frame;
}

class SocketPair {
public:
    int peer = -1;
    int receiver = -1;

    SocketPair() {
        int fds[2];
        requireHarness(socketpair(AF_UNIX, SOCK_STREAM, 0, fds) == 0, "socketpair failed");
        peer = fds[0];
        receiver = fds[1];
        for (int fd : fds)
            requireHarness(fcntl(fd, F_SETFL, fcntl(fd, F_GETFL) | O_NONBLOCK) == 0,
                           "nonblocking socket setup failed");
    }
    ~SocketPair() {
        if (peer != -1) ::close(peer);
        if (receiver != -1) ::close(receiver);
    }

    void writeBytes(const char* bytes, size_t size) {
        while (size) {
            const ssize_t written = ::send(peer, bytes, size, MSG_NOSIGNAL);
            if (written < 0 && errno == EINTR) continue;
            requireHarness(written > 0, std::string("fixture socket write failed: ") + std::strerror(errno));
            bytes += written;
            size -= static_cast<size_t>(written);
        }
    }
    void writeLength(size_t size) {
        const uint64_t length = htobe64(static_cast<uint64_t>(size));
        writeBytes(reinterpret_cast<const char*>(&length), sizeof(length));
    }
    void writeFrame(const Bytes& frame) {
        writeLength(frame.size());
        writeBytes(frame.data(), frame.size());
    }
};

// Forward every request operation to the real MTCL request. Deliver missing
// bytes only AFTER a real test() reports incomplete, preserving that result.
// The next receiver iteration therefore begins with data available and the
// previous request still pending. No thread scheduling or sleeps are involved.
class DeliverBytesAfterIncompletePoll : public MTCL::request_internal {
    MTCL::Request realRequest;
    std::function<void()> deliverBytes;
public:
    DeliverBytesAfterIncompletePoll(MTCL::Request&& request, std::function<void()> deliver)
        : realRequest(std::move(request)), deliverBytes(std::move(deliver)) {}

    int test(int& complete) override {
        complete = MTCL::test(realRequest);
        if (!complete && deliverBytes) {
            auto deliver = std::exchange(deliverBytes, {});
            deliver();
        }
        return 0;
    }
    int wait() override {
        // Also allow a receiver that waits before its first completion poll.
        // A real incomplete test releases the bytes before the blocking wait.
        int complete;
        test(complete);
        return MTCL::wait(realRequest);
    }
    ssize_t count() const override { return realRequest.count(); }
};

// Observe receiver arguments; all socket reads, probes and receives stay real.
class ObservedReceiveHandle : public MTCL::HandleTCP {
public:
    using MTCL::HandleTCP::ireceive;
    size_t receiveCount = 0;
    std::function<void(size_t, size_t)> beforeReceive;
    std::function<void()> afterFirstIncompletePoll;

    ObservedReceiveHandle(MTCL::ConnType* parent, int fd) : MTCL::HandleTCP(parent, fd) {}

    ssize_t ireceive(void* buffer, size_t capacity, MTCL::Request& request) override {
        ++receiveCount;
        if (beforeReceive) beforeReceive(receiveCount, capacity);
        const ssize_t result = MTCL::HandleTCP::ireceive(buffer, capacity, request);
        requireHarness(result == 0, "real MTCL ireceive failed");
        if (receiveCount == 1 && afterFirstIncompletePoll) {
            requireHarness(has_active_receive_req(), "first receive did not remain pending");
            request = MTCL::Request(new DeliverBytesAfterIncompletePoll(
                std::move(request), std::move(afterFirstIncompletePoll)));
        }
        return result;
    }
};

// Publish a socket-backed connection through Manager's normal ready queue.
// Keep ConnTcp's real progress, shutdown and close implementation.
class LocalTcpConnection : public MTCL::ConnTcp {
public:
    inline static LocalTcpConnection* instance = nullptr;
    LocalTcpConnection() { instance = this; }
    void publish(MTCL::HandleTCP* handle) {
        connections[handle->fd] = handle;
        addinQ(true, handle); // performs the real point-to-point handshake
    }
};

class ManagerScope {
public:
    ManagerScope() {
        MTCL::Manager::registerType<LocalTcpConnection>("TEST_SOCKET");
        requireHarness(MTCL::Manager::init("receiver-async-test") == 0, "MTCL init failed");
        ff::MessageAllocator::init(16);
    }
    ~ManagerScope() {
        ff::MessageAllocator::finalize();
        MTCL::Manager::finalize();
    }
};

struct ReceivedMessage {
    std::string payload;
    int source, destination;
    ff::ChannelType type;
    ff::ChannelLocality locality;
};

// Connect the real receiver's output to a local consumer without a farm.
class ReceiverWithOutputCallback : public ff::distributed::receiver::Receiver {
public:
    using ff::distributed::receiver::Receiver::Receiver;
    using ff::distributed::receiver::Receiver::registerCallback;
};

class ReceiverFixture {
    ManagerScope manager;
public:
    SocketPair peer;
    ObservedReceiveHandle* handle;
private:
    // Keep the observed handle alive after svc() destroys its HandleUser.
    MTCL::HandleUser keepAlive;
    std::string endpoint = "unused-local-socket";
    ReceiverWithOutputCallback receiver;
    size_t expectedCount;
public:
    std::vector<ReceivedMessage> received;

    explicit ReceiverFixture(size_t messageCount, ff::distributed::ProtocolOptions options = {})
        : handle(new ObservedReceiveHandle(LocalTcpConnection::instance, peer.receiver)),
          keepAlive(handle, false, false), receiver(endpoint, 1, options), expectedCount(messageCount) {
        // Manager expects one framed native int identifying a P2P connection.
        const int collective = 0;
        peer.writeLength(sizeof(collective));
        peer.writeBytes(reinterpret_cast<const char*>(&collective), sizeof(collective));
        LocalTcpConnection::instance->publish(handle);
        receiver.registerCallback(collectMessage, this);
    }
    ~ReceiverFixture() {
        handle->close(true, true);
        peer.receiver = -1; // ConnTcp closed this descriptor.
    }

    void run() {
        ff::termination_counter.store(1);
        receiver.svc(nullptr); // exercise the real receive loop and handleBatch()
    }

    void expectPayloads(const std::vector<std::string>& payloads) const {
        expect(received.size() == payloads.size(), "wrong number of delivered messages");
        for (size_t i = 0; i < payloads.size(); ++i) {
            const auto& message = received[i];
            expect(message.payload == payloads[i], "payload/order mismatch at message " + std::to_string(i + 1));
            expect(message.source == kSource && message.destination == kDestination &&
                   message.type == ff::ChannelType::FWD && message.locality == ff::ChannelLocality::REMOTE,
                   "metadata mismatch at message " + std::to_string(i + 1));
        }
    }
private:
    static bool collectMessage(void* task, int, unsigned long, unsigned long, void* context) {
        auto& fixture = *static_cast<ReceiverFixture*>(context);
        auto* message = static_cast<ff::message2_t*>(task);
        fixture.received.push_back({std::string(message->data, message->size), message->src,
                                   message->dest, message->type, message->locality});
        ff::MessageAllocator::releaseMessage(message);
        if (fixture.received.size() == fixture.expectedCount)
            ff::termination_counter.store(0);
        return true;
    }
};
} // namespace receiver_test
#endif
