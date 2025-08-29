/* Copyright (c) 2018-2024 Marcelo Zimbres Silva (mzimbres@gmail.com)
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE.txt)
 */

#ifndef BOOST_REDIS_RUNNER_HPP
#define BOOST_REDIS_RUNNER_HPP

#include <boost/redis/config.hpp>
#include <boost/redis/request.hpp>
#include <boost/redis/response.hpp>
#include <boost/redis/error.hpp>
#include <boost/redis/logger.hpp>
#include <boost/redis/operation.hpp>
#include <boost/asio/compose.hpp>
#include <boost/asio/coroutine.hpp>
//#include <boost/asio/ip/tcp.hpp>
#include <string>
#include <memory>
#include <chrono>

namespace boost::redis::detail
{

void push_hello(config const& cfg, request& req);

// TODO: Can we avoid this whole function whose only purpose is to
// check for an error in the hello response and complete with an error
// so that the parallel group that starts it can exit?
template <class Handshaker, class Connection, class Logger>
struct hello_op {
   Handshaker* handshaker_ = nullptr;
   Connection* conn_ = nullptr;
   Logger logger_;
   asio::coroutine coro_{};

   template <class Self>
   void operator()(Self& self, system::error_code ec = {}, std::size_t = 0)
   {
      BOOST_ASIO_CORO_REENTER (coro_)
      {
         handshaker_->add_hello();

         BOOST_ASIO_CORO_YIELD
         conn_->async_exec(handshaker_->hello_req_, any_adapter(handshaker_->hello_resp_), std::move(self));
         logger_.on_hello(ec, handshaker_->hello_resp_);

         if (ec) {
            conn_->cancel(operation::run);
            self.complete(ec);
            return;
         }

         if (handshaker_->has_error_in_response()) {
            conn_->cancel(operation::run);
            self.complete(error::resp3_hello);
            return;
         }

         self.complete({});
      }
   }
};

template <class Executor>
class resp3_handshaker {
public:
   void set_config(config const& cfg)
      { cfg_ = cfg; }

   template <class Connection, class Logger, class CompletionToken>
   auto async_hello(Connection& conn, Logger l, CompletionToken token)
   {
      return asio::async_compose
         < CompletionToken
         , void(system::error_code)
         >(hello_op<resp3_handshaker, Connection, Logger>{this, &conn, l}, token, conn);
   }

private:
   template <class, class, class> friend struct hello_op;

   void add_hello()
   {
      hello_req_.clear();
      if (hello_resp_.has_value())
         hello_resp_.value().clear();
      push_hello(cfg_, hello_req_);
   }

   bool has_error_in_response() const noexcept
   {
      if (!hello_resp_.has_value())
         return true;

      auto f = [](auto const& e)
      {
         switch (e.data_type) {
            case resp3::type::simple_error:
            case resp3::type::blob_error: return true;
            default: return false;
         }
      };

      return std::any_of(std::cbegin(hello_resp_.value()), std::cend(hello_resp_.value()), f);
   }

   request hello_req_;
   generic_response hello_resp_;
   config cfg_;
};

} // boost::redis::detail

#endif // BOOST_REDIS_RUNNER_HPP
