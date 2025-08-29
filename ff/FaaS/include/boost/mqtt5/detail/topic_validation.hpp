//
// Copyright (c) 2023-2025 Ivica Siladic, Bruno Iljazovic, Korina Simicevic
//
// Distributed under the Boost Software License, Version 1.0.
// (See accompanying file LICENSE or copy at http://www.boost.org/LICENSE_1_0.txt)
//

#ifndef BOOST_MQTT5_TOPIC_VALIDATION_HPP
#define BOOST_MQTT5_TOPIC_VALIDATION_HPP

#include <boost/mqtt5/detail/utf8_mqtt.hpp>

#include <cstdint>
#include <string_view>

namespace boost::mqtt5::detail {

static constexpr int32_t min_subscription_identifier = 1;
static constexpr int32_t max_subscription_identifier = 268'435'455;

static constexpr std::string_view shared_sub_prefix = "$share/";

inline bool is_utf8_no_wildcard(validation_result result) {
    return result == validation_result::valid;
}

inline bool is_not_empty(size_t sz) {
    return sz != 0;
}

inline bool is_valid_topic_size(size_t sz) {
    return is_not_empty(sz) && is_valid_string_size(sz);
}

inline validation_result validate_topic_name(std::string_view str) {
    return validate_impl(str, is_valid_topic_size, is_utf8_no_wildcard);
}

inline validation_result validate_topic_alias_name(std::string_view str) {
    return validate_impl(str, is_valid_string_size, is_utf8_no_wildcard);
}

inline validation_result validate_shared_topic_name(std::string_view str) {
    return validate_impl(str, is_not_empty, is_utf8_no_wildcard);
}

inline validation_result validate_topic_filter(std::string_view str) {
    if (!is_valid_topic_size(str.size()))
        return validation_result::invalid;

    constexpr int multi_lvl_wildcard = '#';
    constexpr int single_lvl_wildcard = '+';

    // must be the last character preceded by '/' or stand alone
    // #, .../#
    if (str.back() == multi_lvl_wildcard) {
        str.remove_suffix(1);

        if (!str.empty() && str.back() != '/')
            return validation_result::invalid;
    }

    int last_c = -1;
    validation_result result;
    while (!str.empty()) {
        int c = pop_front_unichar(str);

        // can be used at any level, but must occupy an entire level
        // +, +/..., .../+/..., .../+
        bool is_valid_single_lvl = (c == single_lvl_wildcard) &&
            (str.empty() || str.front() == '/') &&
            (last_c == -1 || last_c == '/');

        result = validate_mqtt_utf8_char(c);
        if (
            result == validation_result::valid ||
            is_valid_single_lvl
        ) {
            last_c = c;
            continue;
        }

        return validation_result::invalid;
    }

    return validation_result::valid;
}

inline validation_result validate_shared_topic_filter(
    std::string_view str, bool wildcard_allowed = true
) {
    if (!is_valid_topic_size(str.size()))
        return validation_result::invalid;

    if (str.compare(0, shared_sub_prefix.size(), shared_sub_prefix) != 0)
        return validation_result::invalid;

    str.remove_prefix(shared_sub_prefix.size());

    size_t share_name_end = str.find_first_of('/');
    if (share_name_end == std::string::npos)
        return validation_result::invalid;

    validation_result result;
    result = validate_shared_topic_name(str.substr(0, share_name_end));

    if (result != validation_result::valid)
        return validation_result::invalid;

    auto topic_filter = str.substr(share_name_end + 1);
    return wildcard_allowed ?
        validate_topic_filter(topic_filter) :
        validate_topic_name(topic_filter)
    ;
}

} // end namespace boost::mqtt5::detail

#endif //BOOST_MQTT5_TOPIC_VALIDATION_HPP
