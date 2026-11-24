// Tests ff_dreceiverMTCL2::svc() with MTCL's TCP handle. Two parts are
// replaced: the accepted connection is a local socket pair published through
// MTCL's Manager, and the receiver passes messages to a callback, not a farm.
//
// The test sends frames from its end of the socket. ObservedReceiveHandle
// reports the capacity of each ireceive and can delay the rest of a frame
// until the receiver finds the first receive pending.
#ifndef FF_TESTS_MTCL_ASYNC_RECEIVER_FIXTURE_HPP
#define FF_TESTS_MTCL_ASYNC_RECEIVER_FIXTURE_HPP

#include "test_support.hpp"

#include <functional>
#include <memory>

namespace mtcl_async_test {

// Wraps an MTCL request. The first test() that finds it pending calls deliver,
// which sends the rest of the frame. The receiver's next iteration then sees
// new data while the first receive is still pending. No threads or sleeps.
class DeliverBytesAfterIncompletePoll : public MTCL::request_internal {
public:
    DeliverBytesAfterIncompletePoll(MTCL::Request&& request, std::function<void()> deliver)
        : request_(std::move(request)), deliver_(std::move(deliver)) {}

    int test(int& complete) override {
        complete = MTCL::test(request_);
        if (!complete && deliver_) std::exchange(deliver_, {})();
        return 0;
    }

    // Calls test() first, so a receiver that waits without testing still gets the bytes.
    int wait() override {
        int complete = 0;
        test(complete);
        return MTCL::wait(request_);
    }

    ssize_t count() const override { return request_.count(); }

private:
    MTCL::Request request_;
    std::function<void()> deliver_;
};

// MTCL's TCP handle with hooks around ireceive.
class ObservedReceiveHandle : public MTCL::HandleTCP {
public:
    using MTCL::HandleTCP::HandleTCP;
    using MTCL::HandleTCP::ireceive;

    // Called before every ireceive with its 1-based number and the buffer capacity.
    void beforeReceive(std::function<void(std::size_t number, std::size_t capacity)> check) {
        beforeReceive_ = std::move(check);
    }

    // Called once, the first time the receiver finds the first receive pending.
    void afterFirstIncompletePoll(std::function<void()> deliver) { afterFirstIncompletePoll_ = std::move(deliver); }

    ssize_t ireceive(void* buffer, std::size_t capacity, MTCL::Request& request) override {
        ++receiveCount_;
        if (beforeReceive_) beforeReceive_(receiveCount_, capacity);
        const ssize_t result = MTCL::HandleTCP::ireceive(buffer, capacity, request);
        require(result == 0, "MTCL ireceive failed");
        if (receiveCount_ == 1 && afterFirstIncompletePoll_) {
            require(has_active_receive_req(), "first receive did not remain pending");
            request = MTCL::Request(new DeliverBytesAfterIncompletePoll(std::move(request),
                                                                        std::exchange(afterFirstIncompletePoll_, {})));
        }
        return result;
    }

private:
    std::size_t receiveCount_ = 0;
    std::function<void(std::size_t, std::size_t)> beforeReceive_;
    std::function<void()> afterFirstIncompletePoll_;
};

// MTCL connection type registered with the Manager. publish() puts a
// test-created handle in the Manager's ready queue, as an accept would.
class LocalReceiverConnection : public MTCL::ConnTcp {
public:
    LocalReceiverConnection() { instance_ = this; }

    [[nodiscard]] static LocalReceiverConnection& instance() {
        require(instance_ != nullptr, "MTCL Manager did not create the test connection type");
        return *instance_;
    }

    void publish(MTCL::HandleTCP& handle) {
        connections[handle.fd] = &handle;
        addinQ(true, &handle); // reads the point-to-point handshake written by the peer
    }

private:
    inline static LocalReceiverConnection* instance_ = nullptr;
};

// MTCL Manager and FastFlow message allocator for the lifetime of one scenario.
class RuntimeScope {
public:
    RuntimeScope() {
        MTCL::Manager::registerType<LocalReceiverConnection>("TEST_SOCKET");
        require(MTCL::Manager::init("receiver-async-test") == 0, "MTCL init failed");
        ff::MessageAllocator::init(kMessagePoolSize);
    }
    ~RuntimeScope() {
        ff::MessageAllocator::finalize();
        MTCL::Manager::finalize();
    }
    RuntimeScope(const RuntimeScope&) = delete;
    RuntimeScope& operator=(const RuntimeScope&) = delete;
    RuntimeScope(RuntimeScope&&) = delete;
    RuntimeScope& operator=(RuntimeScope&&) = delete;

private:
    static constexpr std::size_t kMessagePoolSize = 16;
};

struct ReceivedMessage {
    std::string payload;
    int source;
    int destination;
    ff::ChannelType type;
    ff::ChannelLocality locality;
};

// Exposes the receiver's output callback, so no farm is needed.
class ReceiverWithOutputCallback : public ff::ff_dreceiverMTCL2 {
public:
    using ff::ff_dreceiverMTCL2::ff_dreceiverMTCL2;
    using ff::ff_dreceiverMTCL2::registerCallback;
};

// One receiver, one connection. run() returns after expectedCount messages.
class ReceiverFixture {
public:
    explicit ReceiverFixture(std::size_t expectedCount)
        : sockets_(makeSocketPair()),
          // MTCL owns the handle and deletes it when its last HandleUser goes away;
          // keepAlive_ is that last user, so the handle outlives svc().
          handle_(new ObservedReceiveHandle(&LocalReceiverConnection::instance(), sockets_.mtclEnd.release())),
          keepAlive_(handle_, false, false),
          receiver_(endpoint_, /*input_channels=*/1),
          expectedCount_(expectedCount) {
        // The Manager expects one framed native int marking a point-to-point connection.
        const int pointToPoint = 0;
        sendAll(peer(), mtclLengthPrefix(sizeof(pointToPoint)));
        sendAll(peer(), reinterpret_cast<const char*>(&pointToPoint), sizeof(pointToPoint));
        LocalReceiverConnection::instance().publish(*handle_);
        receiver_.registerCallback(collectMessage, this);
    }
    ~ReceiverFixture() { handle_->close(true, true); }
    ReceiverFixture(const ReceiverFixture&) = delete;
    ReceiverFixture& operator=(const ReceiverFixture&) = delete;
    ReceiverFixture(ReceiverFixture&&) = delete;
    ReceiverFixture& operator=(ReceiverFixture&&) = delete;

    [[nodiscard]] ObservedReceiveHandle& handle() noexcept { return *handle_; }

    // The test's end of the connection.
    [[nodiscard]] int peer() const noexcept { return sockets_.testEnd.get(); }

    void run() {
        ff::termination_counter.store(1);
        receiver_.svc(nullptr); // the receive loop, including handleBatch()
    }

    // Delivered payloads, in order, each with the fixture's source and destination.
    void expectPayloads(const std::vector<std::string>& payloads) const {
        expect(received_.size() == payloads.size(), "wrong number of delivered messages");
        for (std::size_t i = 0; i < payloads.size(); ++i) {
            const auto& message = received_[i];
            const auto position = std::to_string(i + 1);
            expect(message.payload == payloads[i], "payload/order mismatch at message " + position);
            expect(message.source == kSourceId && message.destination == kDestinationId &&
                       message.type == ff::ChannelType::FWD && message.locality == ff::ChannelLocality::REMOTE,
                   "metadata mismatch at message " + position);
        }
    }

private:
    static bool collectMessage(void* task, int, unsigned long, unsigned long, void* context) {
        auto& fixture = *static_cast<ReceiverFixture*>(context);
        auto* message = static_cast<ff::message2_t*>(task);
        fixture.received_.push_back(
            {std::string(message->data, message->size), message->src, message->dest, message->type, message->locality});
        ff::MessageAllocator::releaseMessage(message);
        if (fixture.received_.size() == fixture.expectedCount_) ff::termination_counter.store(0);
        return true;
    }

    RuntimeScope runtime_;           // constructed first, destroyed last
    SocketPair sockets_;
    ObservedReceiveHandle* handle_;  // owned by MTCL, see the constructor
    MTCL::HandleUser keepAlive_;
    std::string endpoint_ = "unused-local-socket";
    ReceiverWithOutputCallback receiver_;
    std::size_t expectedCount_;
    std::vector<ReceivedMessage> received_;
};

} // namespace mtcl_async_test
#endif
