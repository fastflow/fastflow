// Shared support for the MTCL sender and receiver regression tests:
// error categories, socket ownership, the FastFlow wire format and the
// command-line entry point used by both test programs.
#ifndef FF_TESTS_MTCL_ASYNC_TEST_SUPPORT_HPP
#define FF_TESTS_MTCL_ASYNC_TEST_SUPPORT_HPP

// The tests use the sender and receiver classes without starting the distributed runtime.
#define DFF_EXCLUDE_MPI
#define DFF_EXCLUDE_BLOCKING
#include <ff/dff.hpp>

#include <algorithm>
#include <array>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <string_view>
#include <system_error>
#include <type_traits>
#include <utility>
#include <vector>
#include <arpa/inet.h>
#include <endian.h>
#include <fcntl.h>
#include <sys/socket.h>
#include <unistd.h>

namespace mtcl_async_test {

using Bytes = std::vector<char>;

inline constexpr ff::addr_t kSourceId = 3;
inline constexpr ff::addr_t kDestinationId = 7;
inline constexpr ff::addr_t kLogicalEosDestination = -1;

static_assert(sizeof(ff::addr_t) == 2 && sizeof(ff::sizeDFF_t) == 8,
              "Wire fixtures require the 64-bit FastFlow ABI");

// ---------------------------------------------------------------------------
// Failure categories. The runner tells them apart by exit status.

// The code under test produced a wrong result (exit status 1).
struct RegressionFailure : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// The test could not set up its scenario (exit status 2).
struct HarnessError : std::runtime_error {
    using std::runtime_error::runtime_error;
};

inline void expect(bool condition, const std::string& detail) {
    if (!condition) throw RegressionFailure(detail);
}

inline void require(bool condition, const std::string& detail) {
    if (!condition) throw HarnessError(detail);
}

[[noreturn]] inline void throwSystemError(const char* operation) {
    throw std::system_error(errno, std::generic_category(), operation);
}

inline bool wouldBlock(int error) noexcept {
    return error == EAGAIN || error == EWOULDBLOCK;
}

// ---------------------------------------------------------------------------
// Sockets

// Sole owner of a POSIX file descriptor.
class FileDescriptor {
public:
    explicit FileDescriptor(int fd = -1) noexcept : fd_(fd) {}
    ~FileDescriptor() { reset(); }
    FileDescriptor(const FileDescriptor&) = delete;
    FileDescriptor& operator=(const FileDescriptor&) = delete;
    FileDescriptor(FileDescriptor&& other) noexcept : fd_(other.release()) {}
    FileDescriptor& operator=(FileDescriptor&& other) noexcept {
        if (this != &other) {
            reset();
            fd_ = other.release();
        }
        return *this;
    }

    [[nodiscard]] int get() const noexcept { return fd_; }

    // Transfers ownership to the caller, e.g. to an MTCL connection.
    [[nodiscard]] int release() noexcept { return std::exchange(fd_, -1); }

private:
    void reset() noexcept {
        if (fd_ != -1) ::close(fd_);
        fd_ = -1;
    }

    int fd_;
};

// Connected non-blocking stream sockets: MTCL uses one end, the test the other.
struct SocketPair {
    FileDescriptor mtclEnd;
    FileDescriptor testEnd;
};

inline SocketPair makeSocketPair() {
    std::array<int, 2> fds{};
    if (::socketpair(AF_UNIX, SOCK_STREAM, 0, fds.data()) == -1) throwSystemError("socketpair");
    SocketPair sockets{FileDescriptor{fds[0]}, FileDescriptor{fds[1]}};
    for (const int fd : fds) {
        const int flags = ::fcntl(fd, F_GETFL);
        if (flags == -1 || ::fcntl(fd, F_SETFL, flags | O_NONBLOCK) == -1) throwSystemError("fcntl(O_NONBLOCK)");
    }
    return sockets;
}

// Sends every byte. The receiver tests send a few hundred bytes, below the socket buffer size.
inline void sendAll(int fd, const char* data, std::size_t size) {
    while (size > 0) {
        const ssize_t written = ::send(fd, data, size, MSG_NOSIGNAL);
        if (written == -1 && errno == EINTR) continue;
        if (written <= 0) throwSystemError("test socket send");
        data += written;
        size -= static_cast<std::size_t>(written);
    }
}

inline void sendAll(int fd, const Bytes& bytes) { sendAll(fd, bytes.data(), bytes.size()); }

// One recv(), retried on EINTR.
inline ssize_t receiveSome(int fd, char* data, std::size_t size) {
    ssize_t count;
    do {
        count = ::recv(fd, data, size, 0);
    } while (count == -1 && errno == EINTR);
    return count;
}

// ---------------------------------------------------------------------------
// Wire format, encoded here and not by the sender, so a sender bug cannot leak
// into the expected bytes.

template<class T> void append(Bytes& bytes, T value) {
    static_assert(std::is_trivially_copyable_v<T>);
    const auto* raw = reinterpret_cast<const char*>(&value);
    bytes.insert(bytes.end(), raw, raw + sizeof(value));
}

inline void append(Bytes& bytes, const Bytes& more) { bytes.insert(bytes.end(), more.begin(), more.end()); }

// MTCL TCP framing: each frame starts with the 8-byte big-endian length of its body.
inline Bytes mtclLengthPrefix(std::size_t size) {
    Bytes prefix;
    append(prefix, htobe64(static_cast<std::uint64_t>(size)));
    return prefix;
}

// A zero length is MTCL's end of stream, sent when the write side shuts down.
inline Bytes mtclEndOfStream() { return mtclLengthPrefix(0); }

inline Bytes mtclFrame(const Bytes& body) {
    Bytes frame = mtclLengthPrefix(body.size());
    append(frame, body);
    return frame;
}

// A FastFlow batch holding one message, in network byte order:
// [payload][dest:2][src:2][channel:1][payload size:8][message count:4].
// An empty payload addressed to kLogicalEosDestination is a logical EOS.
inline Bytes singleMessageBatch(std::string_view payload, ff::addr_t destination = kDestinationId) {
    Bytes batch(payload.begin(), payload.end());
    append(batch, htons(static_cast<std::uint16_t>(destination)));
    append(batch, htons(static_cast<std::uint16_t>(kSourceId)));
    append(batch, ff::ChannelType::FWD);
    append(batch, htobe64(static_cast<std::uint64_t>(payload.size())));
    append(batch, htonl(std::uint32_t{1}));
    return batch;
}

inline void expectBytesEqual(const Bytes& actual, const Bytes& expected, std::string_view description) {
    const auto mismatch = std::mismatch(actual.begin(), actual.end(), expected.begin(), expected.end());
    if (mismatch.first == actual.end() && mismatch.second == expected.end()) return;
    throw RegressionFailure(std::string(description) + ": first mismatch at byte " +
                            std::to_string(std::distance(actual.begin(), mismatch.first)) + ", expected " +
                            std::to_string(expected.size()) + " bytes, received " + std::to_string(actual.size()));
}

// ---------------------------------------------------------------------------
// Entry point shared by both test programs:
//   <program> --case <name> [--control]
// Without --control the scenario takes the path covered by the fix; with it,
// the scenario avoids that path and passes with or without the fix.

enum class Mode { Regression, Control };

struct Scenario {
    std::string_view name;
    void (*run)(Mode);
};

template<std::size_t N>
int runScenarioFromCommandLine(int argc, char** argv, const std::array<Scenario, N>& scenarios) {
    const auto usage = [&] {
        std::cerr << "usage: " << argv[0] << " --case";
        for (const auto& scenario : scenarios) std::cerr << ' ' << scenario.name;
        std::cerr << " [--control]\n";
        return 2;
    };
    if ((argc != 3 && argc != 4) || std::string_view{argv[1]} != "--case") return usage();
    if (argc == 4 && std::string_view{argv[3]} != "--control") return usage();
    const auto scenario = std::find_if(scenarios.begin(), scenarios.end(),
                                       [&](const Scenario& s) { return s.name == argv[2]; });
    if (scenario == scenarios.end()) return usage();

    try {
        scenario->run(argc == 4 ? Mode::Control : Mode::Regression);
        std::cout << "PASS\n";
        return 0;
    } catch (const RegressionFailure& failure) {
        std::cerr << "FAIL: " << failure.what() << '\n';
        return 1;
    } catch (const std::exception& error) {
        std::cerr << "HARNESS ERROR: " << error.what() << '\n';
        return 2;
    }
}

} // namespace mtcl_async_test
#endif
