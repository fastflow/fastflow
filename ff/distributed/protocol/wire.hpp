// Distributed FastFlow. Licensed under LGPLv3 or MIT, as the enclosing project.
#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <ff/distributed/ff_ddefines.hpp>
#include <ff/distributed/ff_network.hpp>

namespace ff::distributed::wire {

// Preserve the existing wire ABI. Fixed-width/versioned records are a separate
// protocol change; in particular sizeDFF_t still depends on the host ABI.
constexpr std::size_t recordHeaderBytes = 2 * sizeof(addr_t) + sizeof(ChannelType) + sizeof(sizeDFF_t);
constexpr std::size_t batchCountBytes = sizeof(int);
static_assert(sizeof(int) == sizeof(std::uint32_t), "wire batch count requires a 32-bit int");

struct RecordHeader {
    int destination;
    int source;
    ChannelType channel;
    sizeDFF_t payloadBytes;
};

template<class T> inline T networkOrder(T value) {
    static_assert(std::is_integral<T>::value, "integral wire field required");
    if constexpr (sizeof(T) == 2) return static_cast<T>(htons(static_cast<std::uint16_t>(value)));
    if constexpr (sizeof(T) == 4) return static_cast<T>(htonl(static_cast<std::uint32_t>(value)));
    if constexpr (sizeof(T) == 8) return static_cast<T>(htobe64(static_cast<std::uint64_t>(value)));
    return value;
}

template<class T> inline T hostOrder(T value) {
    if constexpr (sizeof(T) == 2) return static_cast<T>(ntohs(static_cast<std::uint16_t>(value)));
    if constexpr (sizeof(T) == 4) return static_cast<T>(ntohl(static_cast<std::uint32_t>(value)));
    if constexpr (sizeof(T) == 8) return static_cast<T>(be64toh(static_cast<std::uint64_t>(value)));
    return value;
}

// The packed layout contains unaligned fields. memcpy deliberately avoids
// dereferencing cast pointers; encode and decode share the same field offsets.
template<class T> inline void store(char*& destination, T value) {
    value = networkOrder(value);
    std::memcpy(destination, &value, sizeof(value));
    destination += sizeof(value);
}

template<class T> inline T load(const char*& source) {
    T value;
    std::memcpy(&value, source, sizeof(value));
    source += sizeof(value);
    return hostOrder(value);
}

inline addr_t address(int value) {
    if (value < -1 || value > std::numeric_limits<addr_t>::max())
        throw std::invalid_argument("node address exceeds the wire ABI");
    return static_cast<addr_t>(value);
}

inline RecordHeader describe(const message2_t& message) {
    if (message.size && !message.data)
        throw std::invalid_argument("nonempty message has no payload");
    address(message.dest);
    address(message.src);
    return {message.dest, message.src,
            message.size ? message.type : ChannelType::FWD, message.size};
}

inline void encodeHeader(char* destination, const RecordHeader& record) {
    store(destination, address(record.destination));
    store(destination, address(record.source));
    std::memcpy(destination, &record.channel, sizeof(record.channel));
    destination += sizeof(record.channel);
    store(destination, record.payloadBytes);
}

inline RecordHeader decodeHeader(const char* source) {
    RecordHeader record;
    record.destination = load<addr_t>(source);
    record.source = load<addr_t>(source);
    std::memcpy(&record.channel, source, sizeof(record.channel));
    source += sizeof(record.channel);
    record.payloadBytes = load<sizeDFF_t>(source);
    if (record.destination < -1 || record.source < 0 ||
        (record.payloadBytes && record.channel != ChannelType::FWD && record.channel != ChannelType::FBK))
        throw std::runtime_error("invalid distributed record header");
    return record;
}

inline void encodeCount(char* destination, std::size_t count) {
    if (!count || count > static_cast<std::size_t>(std::numeric_limits<int>::max()))
        throw std::invalid_argument("invalid distributed batch count");
    store(destination, static_cast<std::uint32_t>(count));
}

inline std::size_t decodeCount(const char* source) {
    return load<std::uint32_t>(source);
}

} // namespace ff::distributed::wire
