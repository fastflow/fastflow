// Regressions for stale frame sizes and stale buffer capacities in the receiver.
// Low-level socket/MTCL setup lives in dreceiverMTCL_async_test_support.hpp.
#if defined(__has_feature)
#if __has_feature(address_sanitizer)
#define RECEIVER_TEST_HAS_ASAN 1
#endif
#endif
#if !defined(__SANITIZE_ADDRESS__) && !defined(RECEIVER_TEST_HAS_ASAN)
#error "Compile this test with -fsanitize=address to detect buffer-capacity regressions"
#endif
#include "dreceiverMTCL_async_test_support.hpp"

using namespace receiver_test;

namespace {
enum class Delivery { CompleteFrames, FirstFrameSplit };
enum class BufferSequence { GrowAfterShrink, StaySmallAfterShrink };

void testNextReceiveUsesTheNextFrameSize(Delivery delivery) {
    // Arrange: the second frame is larger, so reusing the first size is invalid.
    ReceiverFixture fixture(2);
    const std::string firstPayload(32, 'A');
    const std::string secondPayload(128, 'B');
    const Bytes firstFrame = messageFrame(firstPayload);
    const Bytes secondFrame = messageFrame(secondPayload);
    bool deliveredAfterIncompletePoll = false;

    fixture.handle->beforeReceive = [&](size_t number, size_t capacity) {
        if (number == 2) {
            std::cout << "frame-size: second receive capacity=" << capacity
                      << ", next frame requires=" << secondFrame.size() << " bytes\n";
            // Fail before MTCL drains the larger frame into a too-small buffer.
            // Final payload assertions also catch wrong parsing or ordering.
            expect(capacity >= secondFrame.size(),
                   "receiver reused the previous frame size: second receive has " +
                   std::to_string(capacity) + " bytes, needs " + std::to_string(secondFrame.size()));
        }
    };

    if (delivery == Delivery::CompleteFrames) {
        fixture.peer.writeFrame(firstFrame);
        fixture.peer.writeFrame(secondFrame);
    } else {
        // Publish only A's length and its first byte: ireceive and the first
        // completion poll must both report that A is still pending.
        fixture.peer.writeLength(firstFrame.size());
        fixture.peer.writeBytes(firstFrame.data(), 1);
        fixture.handle->afterFirstIncompletePoll = [&] {
            fixture.peer.writeBytes(firstFrame.data() + 1, firstFrame.size() - 1);
            fixture.peer.writeFrame(secondFrame);
            deliveredAfterIncompletePoll = true;
        };
    }

    // Act: run the real receiver. It must finish A before probing for B's size.
    fixture.run();

    // Assert: both messages arrive intact, in order, with their own metadata.
    if (delivery == Delivery::FirstFrameSplit)
        requireHarness(deliveredAfterIncompletePoll, "split-frame gate was never exercised");
    fixture.expectPayloads({firstPayload, secondPayload});
}

void testReceiveBufferCapacityTracksItsActualAllocation(BufferSequence sequence) {
    // Arrange: A grows slot 0; B transfers that allocation to the consumer and
    // replaces it with a smaller allocation. C must fit the replacement buffer.
    // Regression: receive 256 payload bytes, then 32, then 128 (grow after shrinking).
    // Control: receive 256 payload bytes, then 32, then 32 (replacement is sufficient).
    ReceiverFixture fixture(3);
    const std::string largePayload(256, 'A');
    const std::string smallPayload(32, 'B');
    const size_t nextPayloadSize = sequence == BufferSequence::StaySmallAfterShrink ? 32 : 128;
    const std::string nextPayload(nextPayloadSize, 'C');
    fixture.peer.writeFrame(messageFrame(largePayload));
    fixture.peer.writeFrame(messageFrame(smallPayload));
    fixture.peer.writeFrame(messageFrame(nextPayload));

    std::cout << "buffer-capacity: first payload=256 bytes, second payload=32 bytes, third payload="
              << nextPayload.size() << " bytes\n";
    std::cout.flush(); // preserve the scenario description if ASan aborts

    // Act: complete frames avoid pending receives, isolating this from the
    // frame-size bug. ASan checks the actual write into the replacement buffer.
    fixture.run();

    // Assert: all payloads and metadata survive the buffer ownership transfers.
    fixture.expectPayloads({largePayload, smallPayload, nextPayload});
}
} // namespace

int main(int argc, char** argv) {
    if ((argc != 3 && argc != 4) || std::string(argv[1]) != "--case" ||
        (std::string(argv[2]) != "frame-size" && std::string(argv[2]) != "buffer-capacity") ||
        (argc == 4 && std::string(argv[3]) != "--control")) {
        std::cerr << "usage: " << argv[0] << " --case frame-size|buffer-capacity [--control]\n";
        return 2;
    }
    const bool control = argc == 4;
    try {
        if (std::string(argv[2]) == "frame-size")
            testNextReceiveUsesTheNextFrameSize(control ? Delivery::CompleteFrames : Delivery::FirstFrameSplit);
        else
            testReceiveBufferCapacityTracksItsActualAllocation(
                control ? BufferSequence::StaySmallAfterShrink : BufferSequence::GrowAfterShrink);
        std::cout << "PASS\n";
        return 0;
    } catch (const RegressionFailure& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "HARNESS ERROR: " << error.what() << '\n';
        return 2;
    }
}
