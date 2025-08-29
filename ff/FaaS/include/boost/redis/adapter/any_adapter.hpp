/* Copyright (c) 2018-2023 Marcelo Zimbres Silva (mzimbres@gmail.com)
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE.txt)
 */

#ifndef BOOST_REDIS_ANY_ADAPTER_HPP
#define BOOST_REDIS_ANY_ADAPTER_HPP


#include <boost/redis/resp3/node.hpp>
#include <boost/redis/adapter/adapt.hpp>
#include <boost/system/error_code.hpp>
#include <cstddef>
#include <functional>
#include <string_view>
#include <type_traits>

namespace boost::redis {

namespace detail { 

// Forward decl
template <class Executor>
class basic_connection;

}

/** @brief A type-erased reference to a response.
 *  @ingroup high-level-api
 *
 *  A type-erased response adapter. It can be executed using @ref connection::async_exec.
 *  Using this type instead of raw response references enables separate compilation.
 *
 *  Given a response object `resp` that can be passed to `async_exec`, the following two
 *  statements have the same effect:
 *  ```
 *      co_await conn.async_exec(req, resp);
 *      co_await conn.async_exec(req, any_response(resp));
 *  ```
 */
class any_adapter
{
    using fn_type = std::function<void(std::size_t, resp3::basic_node<std::string_view> const&, system::error_code&)>;

    struct impl_t {
        fn_type adapt_fn;
        std::size_t supported_response_size;
    } impl_;

    template <class T>
    static auto create_impl(T& resp) -> impl_t
    {
        using namespace boost::redis::adapter;
        auto adapter = boost_redis_adapt(resp);
        std::size_t size = adapter.get_supported_response_size();
        return { std::move(adapter), size };
    }

    template <class Executor>
    friend class basic_connection;

public:
    /**
     * @brief Constructor.
     * 
     * Creates a type-erased response adapter from `resp` by calling
     * `boost_redis_adapt`. `T` must be a valid Redis response type.
     * Any type passed to @ref connection::async_exec qualifies.
     *
     * This object stores a reference to `resp`, which must be kept alive
     * while `*this` is being used.
     */
    template <class T, class = std::enable_if_t<!std::is_same_v<T, any_adapter>>>
    explicit any_adapter(T& resp) : impl_(create_impl(resp)) {}
};

}

#endif
