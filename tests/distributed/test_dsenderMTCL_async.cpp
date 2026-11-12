// Exercise the real sender and MTCL transport, with and without backpressure.
#include "dsenderMTCL_async_test_support.hpp"

#include <csignal>
#include <string_view>

namespace {
using namespace sender_test;

enum class SendMode { ImmediatelyWritable, ForcedPending };

void testCloseWaitsForPendingSends(SendMode mode) {
    TestTransport transport;
    constexpr int messagesPerBatch = 1;
    SenderWithOneConnection sender(transport.handle(), messagesPerBatch, kBatchByteLimit);
    const std::string payload(kSmallPayloadBytes, 'A');
    if (mode == SendMode::ForcedPending) transport.fillSendQueue();

    sender.svc(makeDataMessage(payload).release());
    sender.svc(makeLogicalEos().release());

    Bytes received;
    if (mode == SendMode::ImmediatelyWritable) {
        completePendingSendsAndCollectWire(transport, sender, received);
    } else {
        requireHarness(transport.handle().firstSendWasPending() &&
                           transport.handle().has_active_send_req(),
                       "first send did not remain pending under socket pressure");
        transport.drainQueueFiller();
        // Draining the peer does not progress MTCL: svc_end() must finish
        // the pending requests before shutting down writes.
    }

    sender.svc_end();

    requireHarness(transport.collectReceivedBytes(received) == ReceiveState::PeerClosed,
                   "svc_end did not shut down the write side");
    Bytes expected;
    appendExpectedMessageFrame(expected, payload);
    appendExpectedMessageFrame(expected, "", -1); // FastFlow logical EOS
    append(expected, std::uint64_t{0});           // MTCL transport EOS
    expectWireEquals(received, expected, "data and logical EOS must precede transport EOS");
}

void testPendingBatchBufferIsNotOverwritten(SendMode mode) {
    TestTransport transport;
    constexpr int messagesPerBatch = 8;
    SenderWithOneConnection sender(transport.handle(), messagesPerBatch, kBatchByteLimit);
    const std::string batchedPayload(kSmallPayloadBytes, 'A');
    const std::string directPayload(kBatchByteLimit, 'B');

    if (mode == SendMode::ForcedPending) {
        transport.fillSendQueue();
        transport.handle().afterFirstSend([&transport] {
            requireHarness(transport.handle().firstSendWasPending(),
                           "batch flush did not remain pending under socket pressure");
            // Release pressure without completing A's request. B's direct
            // send must preserve A's buffer until MTCL finishes reading it.
            transport.drainQueueFiller();
        });
    }
    sender.svc(makeDataMessage(batchedPayload).release());
    requireHarness(transport.handle().sendCallCount() == 0, "small message was not batched");

    // B reaches the byte limit, flushing A before taking the direct-send path.
    sender.svc(makeDataMessage(directPayload).release());

    Bytes received;
    completePendingSendsAndCollectWire(transport, sender, received);
    sender.svc_end();
    requireHarness(transport.collectReceivedBytes(received) == ReceiveState::PeerClosed,
                   "svc_end did not shut down the write side");
    Bytes expected;
    appendExpectedMessageFrame(expected, batchedPayload);
    appendExpectedMessageFrame(expected, directPayload);
    append(expected, std::uint64_t{0});
    expectWireEquals(received, expected, "batched A must arrive intact before direct B");
}
} // namespace

int main(int argc, char** argv) {
    using std::string_view;
    if ((argc != 3 && argc != 4) || string_view{argv[1]} != "--case" ||
        (string_view{argv[2]} != "close" && string_view{argv[2]} != "overwrite") ||
        (argc == 4 && string_view{argv[3]} != "--control")) {
        std::cerr << "usage: " << argv[0] << " --case close|overwrite [--control]\n";
        return 2;
    }

    std::signal(SIGPIPE, SIG_IGN);
    try {
        sender_test::MessagePool messages;
        const auto mode = argc == 4 ? SendMode::ImmediatelyWritable : SendMode::ForcedPending;
        if (string_view{argv[2]} == "close") testCloseWaitsForPendingSends(mode);
        else testPendingBatchBufferIsNotOverwritten(mode);
        std::cout << "PASS\n";
        return 0;
    } catch (const sender_test::RegressionFailure& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "HARNESS ERROR: " << error.what() << '\n';
        return 2;
    }
}
