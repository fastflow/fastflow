// Regression tests for asynchronous sends in ff_dsenderMTCL2.
//
//   close      svc_end() must complete pending sends before shutting down writes.
//   overwrite  a direct send after a batch flush must not reuse the flushed
//              buffer while its send is still pending.
//
// Regression mode fills the socket send buffer, so the first send stays
// pending. Control mode leaves room, so every send completes at once.
#include "sender_fixture.hpp"

#include <csignal>

namespace {
using namespace mtcl_async_test;

void testCloseCompletesPendingSends(Mode mode) {
    MessagePool messages;
    TestTransport transport;
    SenderWithOneConnection sender(transport.handle(), /*batchCount=*/1, kBatchByteLimit);
    const std::string payload(kSmallPayloadBytes, 'A');
    if (mode == Mode::Regression) transport.fillSendQueue();

    sender.svc(makeDataMessage(payload).release());
    sender.svc(makeLogicalEos().release());

    Bytes wire;
    if (mode == Mode::Control) {
        completePendingSends(transport, wire);
    } else {
        require(transport.handle().firstSendWasPending() && transport.handle().has_active_send_req(),
                "first send did not stay pending with a full send buffer");
        // Reading the filler does not advance MTCL: svc_end() must complete
        // the pending sends before it shuts down writes.
        transport.drainSendQueue();
    }
    sender.svc_end();

    require(transport.collectWire(wire) == ReceiveState::PeerClosed, "svc_end did not shut down the write side");
    Bytes expected = expectedMessageFrame(payload);
    append(expected, expectedMessageFrame("", kLogicalEosDestination));
    append(expected, mtclEndOfStream());
    expectBytesEqual(wire, expected, "data and logical EOS must precede the transport EOS");
}

void testPendingBatchBufferIsNotOverwritten(Mode mode) {
    MessagePool messages;
    TestTransport transport;
    SenderWithOneConnection sender(transport.handle(), /*batchCount=*/8, kBatchByteLimit);
    const std::string batchedPayload(kSmallPayloadBytes, 'A');
    const std::string directPayload(kBatchByteLimit, 'B');

    if (mode == Mode::Regression) {
        transport.fillSendQueue();
        transport.handle().afterFirstSend([&transport] {
            require(transport.handle().firstSendWasPending(), "batch flush did not stay pending with a full send buffer");
            // Empty the send buffer without completing A's send: B's direct
            // send must not touch A's buffer before MTCL has written it.
            transport.drainSendQueue();
        });
    }
    sender.svc(makeDataMessage(batchedPayload).release());
    require(transport.handle().sendCallCount() == 0, "small message was not batched");

    // B reaches the byte limit: the sender flushes A, then sends B directly.
    sender.svc(makeDataMessage(directPayload).release());

    Bytes wire;
    completePendingSends(transport, wire);
    sender.svc_end();

    require(transport.collectWire(wire) == ReceiveState::PeerClosed, "svc_end did not shut down the write side");
    Bytes expected = expectedMessageFrame(batchedPayload);
    append(expected, expectedMessageFrame(directPayload));
    append(expected, mtclEndOfStream());
    expectBytesEqual(wire, expected, "batched A must arrive intact before direct B");
}

constexpr std::array<Scenario, 2> kScenarios{{
    {"close", testCloseCompletesPendingSends},
    {"overwrite", testPendingBatchBufferIsNotOverwritten},
}};
} // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN); // a closed peer must surface as an error, not end the process
    return runScenarioFromCommandLine(argc, argv, kScenarios);
}
