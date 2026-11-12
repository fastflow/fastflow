// Contract tests use independent wire fixtures and real MTCL TCP handles.
#include "dsenderMTCL_async_test_support.hpp"
#include "dreceiverMTCL_async_test_support.hpp"

#include <array>
#include <csignal>
#include <iostream>
#include <limits>
#include <string_view>

namespace {
namespace distributed = ff::distributed;

void expect(bool condition, const char* detail) {
    if (!condition) throw std::runtime_error(detail);
}

template<class Exception, class Action> void expectFailure(Action action, const char* detail) {
    try {
        action();
    } catch (const Exception&) {
        return;
    }
    throw std::runtime_error(detail);
}

void testProtocolValidation() {
    // The codec must support the actual packed ABI, including unaligned headers.
    std::array<char, distributed::wire::recordHeaderBytes + 1> storage{};
    distributed::wire::encodeHeader(storage.data() + 1, {-1, 3, ff::FWD, 32});
    const auto header = distributed::wire::decodeHeader(storage.data() + 1);
    expect(header.destination == -1 && header.source == 3 && header.payloadBytes == 32,
           "packed record metadata changed");

    const auto good = receiver_test::messageFrame("data");
    const auto parsed = distributed::receiver::decodeBatch(good.data(), good.size(), 64);
    expect(parsed.records.size() == 1 && parsed.records.front().payloadBytes == 4,
           "independent frame was not decoded");
    expectFailure<std::runtime_error>([&] {
        distributed::receiver::decodeBatch(good.data(), 2, 64);
    }, "truncated frame was accepted");

    auto inconsistent = good;
    // Inflate the declared payload without adding bytes to the independent frame.
    const std::uint64_t inflated = htobe64(40);
    std::memcpy(inconsistent.data() + 4 + 2 * sizeof(ff::addr_t) + sizeof(ff::ChannelType),
                &inflated, sizeof(inflated));
    expectFailure<std::runtime_error>([&] {
        distributed::receiver::decodeBatch(inconsistent.data(), inconsistent.size(), 64);
    }, "out-of-bounds payload was accepted");

    auto emptyCount = good;
    std::fill(emptyCount.end() - sizeof(int), emptyCount.end(), 0);
    expectFailure<std::runtime_error>([&] {
        distributed::receiver::decodeBatch(emptyCount.data(), emptyCount.size(), 64);
    }, "zero record count was accepted");
    expectFailure<std::invalid_argument>([] {
        distributed::wire::address(std::numeric_limits<ff::addr_t>::max() + 1);
    }, "an overflowing node address was accepted");

    distributed::SendTokens tokens(distributed::FlowControl::Tokens, 1);
    tokens.acquire();
    expect(!tokens.available() && tokens.outstanding() == 1, "token acquisition changed the invariant");
    expectFailure<std::logic_error>([&] { tokens.acquire(); }, "token window allowed overcommit");
    tokens.confirm();
    expectFailure<std::runtime_error>([&] { tokens.confirm(); }, "duplicate confirmation was accepted");
}

void testTokensAndRouting() {
    sender_test::MessagePool messages;
    sender_test::TestTransport first;
    sender_test::TestTransport second;
    distributed::SenderOptions options;
    options.protocol.flowControl = distributed::FlowControl::Tokens;
    options.sendTokens = 1;
    std::vector<std::unique_ptr<distributed::sender::Session>> sessions;
    sessions.push_back(std::make_unique<distributed::sender::Session>(
        MTCL::HandleUser(&first.handle(), true, false), options));
    sessions.push_back(std::make_unique<distributed::sender::Session>(
        MTCL::HandleUser(&second.handle(), true, false), options));
    const auto message = sender_test::makeDataMessage("payload");
    for (const auto& session : sessions) {
        session->sendSingle(*message, options.protocol.separatePayloadAbove);
        session->completeSends();
        expect(!session->hasToken(), "MTCL completion returned a send token");
    }

    distributed::sender::Selection selection(distributed::LoadBalancing::RoundRobin);
    const std::set<distributed::sender::SessionId> allowed{0};
    second.sendConfirmation();
    sessions[1]->pollConfirmation();
    expect(sessions[1]->hasToken(), "confirmation did not return a send token");
    expect(!selection.choose(allowed, sessions), "selection escaped the source's permitted sessions");
    first.sendConfirmation();
    sessions[0]->pollConfirmation();
    expect(selection.choose(allowed, sessions) == std::optional<distributed::sender::SessionId>{0},
           "eligible session was not selected after confirmation");

    first.sendConfirmation();
    expectFailure<std::runtime_error>([&] { sessions[0]->pollConfirmation(); },
                                     "unsolicited wire confirmation was accepted");
    for (const auto& session : sessions) session->closeWrites();
}

void testBatchDelivery() {
    receiver_test::ReceiverFixture fixture(2);
    const std::string first = "first";
    const std::string second = "second";
    const auto firstFrame = receiver_test::messageFrame(first);
    const auto secondFrame = receiver_test::messageFrame(second);
    receiver_test::Bytes batch(first.begin(), first.end());
    batch.insert(batch.end(), second.begin(), second.end());
    batch.insert(batch.end(), firstFrame.begin() + first.size(), firstFrame.end() - sizeof(int));
    batch.insert(batch.end(), secondFrame.begin() + second.size(), secondFrame.end() - sizeof(int));
    receiver_test::append(batch, htonl(std::uint32_t{2}));
    fixture.peer.writeFrame(batch);
    fixture.run();
    fixture.expectPayloads({first, second});
}

void testSeparatePayload() {
    distributed::ProtocolOptions options;
    options.separatePayloadAbove = 64;
    receiver_test::ReceiverFixture fixture(2, options);
    const std::string payload(128, 'A');
    auto descriptor = receiver_test::messageFrame(payload);
    descriptor.erase(descriptor.begin(), descriptor.begin() + payload.size());
    fixture.peer.writeFrame(descriptor);
    fixture.peer.writeFrame(receiver_test::Bytes(payload.begin(), payload.end()));
    fixture.peer.writeFrame(receiver_test::messageFrame("next batch"));
    fixture.run();
    fixture.expectPayloads({payload, "next batch"});
    expect(fixture.handle->receiveCount == 3, "separate payload consumed the next batch");
}

void testAddressedProducerEnd() {
    receiver_test::ReceiverFixture fixture(2);
    fixture.peer.writeFrame(receiver_test::messageFrame("data"));
    fixture.peer.writeFrame(receiver_test::messageFrame(""));
    fixture.run();
    fixture.expectPayloads({"data", ""});
}

void testFinalBatchDrain() {
    sender_test::MessagePool messages;
    sender_test::TestTransport transport;
    sender_test::SenderWithOneConnection sender(transport.handle(), 8, 128);
    sender.svc(sender_test::makeDataMessage("last batch").release());
    expect(transport.handle().sendCallCount() == 0, "partial batch was not accumulated");
    sender.svc_end();
    sender_test::Bytes received;
    expect(transport.collectReceivedBytes(received) == sender_test::ReceiveState::PeerClosed,
           "sender did not close after draining the partial batch");
    sender_test::Bytes expected;
    sender_test::appendExpectedMessageFrame(expected, "last batch");
    sender_test::append(expected, std::uint64_t{0});
    sender_test::expectWireEquals(received, expected, "partial batch was lost at shutdown");
}
} // namespace

int main(int argc, char** argv) {
    std::signal(SIGPIPE, SIG_IGN);
    try {
        expect(argc == 2, "choose protocol, tokens, batch, separate, producer-end or drain");
        const std::string_view scenario(argv[1]);
        if (scenario == "protocol") testProtocolValidation();
        else if (scenario == "tokens") testTokensAndRouting();
        else if (scenario == "batch") testBatchDelivery();
        else if (scenario == "separate") testSeparatePayload();
        else if (scenario == "producer-end") testAddressedProducerEnd();
        else if (scenario == "drain") testFinalBatchDrain();
        else throw std::runtime_error("unknown transport module scenario");
        std::cout << "PASS " << scenario << '\n';
        return 0;
    } catch (const std::exception& exception) {
        std::cerr << "FAIL: " << exception.what() << '\n';
        return 1;
    }
}
