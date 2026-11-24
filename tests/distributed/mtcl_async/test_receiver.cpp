// Regression tests for asynchronous receives in ff_dreceiverMTCL2.
//
//   frame-size       a second receive on a connection must not reuse the size
//                    of a frame that is still arriving.
//                    Regression: the first frame arrives split.
//                    Control: both frames arrive complete.
//   buffer-capacity  after a receive buffer is handed to a message and
//                    replaced, its recorded capacity must match the new one.
//                    Regression: payloads of 256, 32, then 128 bytes.
//                    Control: payloads of 256, 32, then 32 bytes.
//
// AddressSanitizer detects the out-of-bounds write of buffer-capacity.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RECEIVER_TEST_HAS_ASAN 1
#endif
#endif
#if !defined(__SANITIZE_ADDRESS__) && !defined(RECEIVER_TEST_HAS_ASAN)
#error "Compile this test with -fsanitize=address to detect buffer-capacity regressions"
#endif

#include "receiver_fixture.hpp"

namespace {
using namespace mtcl_async_test;

void testNextReceiveUsesTheNextFrameSize(Mode mode) {
    ReceiverFixture fixture(/*expectedCount=*/2);
    const std::string firstPayload(32, 'A');
    const std::string secondPayload(128, 'B');   // a receive sized for the first cannot hold it
    const Bytes firstBatch = singleMessageBatch(firstPayload);
    const Bytes secondBatch = singleMessageBatch(secondPayload);
    bool splitDeliveryHappened = false;

    fixture.handle().beforeReceive([&](std::size_t number, std::size_t capacity) {
        if (number != 2) return;
        std::cout << "frame-size: second receive capacity=" << capacity
                  << ", next frame requires=" << secondBatch.size() << " bytes\n";
        // Fail before MTCL writes the larger frame into the smaller buffer.
        expect(capacity >= secondBatch.size(), "receiver reused the previous frame size: second receive has " +
                                                   std::to_string(capacity) + " bytes, needs " +
                                                   std::to_string(secondBatch.size()));
    });

    if (mode == Mode::Control) {
        sendAll(fixture.peer(), mtclFrame(firstBatch));
        sendAll(fixture.peer(), mtclFrame(secondBatch));
    } else {
        // Only the length and the first byte of A: its receive must stay pending.
        sendAll(fixture.peer(), mtclLengthPrefix(firstBatch.size()));
        sendAll(fixture.peer(), firstBatch.data(), 1);
        fixture.handle().afterFirstIncompletePoll([&] {
            sendAll(fixture.peer(), firstBatch.data() + 1, firstBatch.size() - 1);
            sendAll(fixture.peer(), mtclFrame(secondBatch));
            splitDeliveryHappened = true;
        });
    }

    fixture.run();

    if (mode == Mode::Regression) require(splitDeliveryHappened, "split delivery was never exercised");
    fixture.expectPayloads({firstPayload, secondPayload});
}

void testReceiveBufferCapacityTracksItsAllocation(Mode mode) {
    // A grows the receive buffer; B's buffer goes to the consumer and is
    // replaced by a smaller one; C must fit that replacement.
    ReceiverFixture fixture(/*expectedCount=*/3);
    const std::string largePayload(256, 'A');
    const std::string smallPayload(32, 'B');
    const std::string nextPayload(mode == Mode::Control ? 32 : 128, 'C');
    for (const auto* payload : {&largePayload, &smallPayload, &nextPayload})
        sendAll(fixture.peer(), mtclFrame(singleMessageBatch(*payload)));

    std::cout << "buffer-capacity: payloads of 256, 32, then " << nextPayload.size() << " bytes\n";
    std::cout.flush(); // keep the description if AddressSanitizer aborts

    // Every frame is complete, so no receive stays pending and frame-size
    // cannot interfere.
    fixture.run();

    fixture.expectPayloads({largePayload, smallPayload, nextPayload});
}

constexpr std::array<Scenario, 2> kScenarios{{
    {"frame-size", testNextReceiveUsesTheNextFrameSize},
    {"buffer-capacity", testReceiveBufferCapacityTracksItsAllocation},
}};
} // namespace

int main(int argc, char** argv) { return runScenarioFromCommandLine(argc, argv, kScenarios); }
