#ifndef FF_FAAS_TYPETRAITS_HPP
#define FF_FAAS_TYPETRAITS_HPP

#include <bitsery/bitsery.h>
#include <bitsery/adapter/measure_size.h>
#include <ff/distributed/ff_typetraits.hpp>

namespace ff {
namespace traits {
    template <typename T, typename = void>
    struct is_bitsery_serializable : std::false_type {};

    template<typename T>
    inline constexpr bool always_false_v = false;

    template <typename T>
    struct is_bitsery_serializable<T, std::void_t<decltype(bitsery::quickSerialization<bitsery::MeasureSize>(
        std::declval<bitsery::MeasureSize>(), std::declval<T&>()))>> : std::true_type {};

    template <typename T, typename = void>
    struct is_bitsery_deserializable : std::false_type {};

    template <typename T>
    struct is_bitsery_deserializable<T, std::void_t<decltype(bitsery::quickDeserialization<bitsery::MeasureSize>(
        std::declval<bitsery::MeasureSize>(), std::declval<T&>()))>> : std::true_type {};
    }
} // namespace ff

#endif // FF_FAAS_TYPETRAITS_HPP