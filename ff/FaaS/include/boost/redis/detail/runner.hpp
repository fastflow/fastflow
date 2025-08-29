/* Copyright (c) 2018-2024 Marcelo Zimbres Silva (mzimbres@gmail.com)
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE.txt)
 */

#ifndef BOOST_REDIS_RUNNER_HPP
#define BOOST_REDIS_RUNNER_HPP

#include <boost/redis/adapter/any_adapter.hpp>
#include <boost/redis/config.hpp>
#include <boost/redis/request.hpp>
#include <boost/redis/response.hpp>
#include <boost/redis/detail/helper.hpp>
#include <boost/redis/error.hpp>
#include <boost/redis/logger.hpp>
#include <boost/redis/operation.hpp>
#include <boost/asio/compose.hpp>
#include <boost/asio/coroutine.hpp>
#include <boost/asio/experimental/parallel_group.hpp>
#include <boost/asio/ip/tcp.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/prepend.hpp>
#include <boost/asio/ssl.hpp>
#include <boost/asio/cancel_after.hpp>
#include <string>
#include <memory>
#include <chrono>

namespace boost::redis::detail
{

void push_hello(config const& cfg, request& req);

// TODO: Can we avoid this whole function whose only purpose is to
// check for an error in the hello response and complete with an error
// so that the parallel group that starts it can exit?
template <class Runner, class Connection, class Logger>
struct hello_op {
   Runner* runner_ = nullptr;
   Connection* conn_ = nullptr;
   Logger logger_;
   asio::coroutine coro_{};

   template <class Self>
   void operator()(Self& self, system::error_code ec = {}, std::size_t = 0)
   {
      BOOST_ASIO_CORO_REENTER (coro_)
      {
         runner_->add_hello();

         BOOST_ASIO_CORO_YIELD
         conn_->async_exec(runner_->hello_req_, any_adapter(runner_->hello_resp_), std::move(self));
         logger_.on_hello(ec, runner_->hello_resp_);

         if (ec) {
            conn_->cancel(operation::run);
            self.complete(ec);
            return;
         }

         if (runner_->has_error_in_response()) {
            conn_->cancel(operation::run);
            self.complete(error::resp3_hello);
            return;
         }

         self.complete({});
      }
   }
};

template <class Runner, class Connection, class Logger>
class runner_op {
private:
   Runner* runner_ = nullptr;
   Connection* conn_ = nullptr;
   Logger logger_;
   asio::coroutine coro_{};

   using order_t = std::array<std::size_t, 5>;

public:
   runner_op(Runner* runner, Connection* conn, Logger l)
   : runner_{runner}
   , conn_{conn}
   , logger_{l}
   {}

   template <class Self>
   void operator()( Self& self
                  , order_t order = {}
                  , system::error_code ec0 = {}
                  , system::error_code ec1 = {}
                  , system::error_code ec2 = {}
                  , system::error_code ec3 = {}
                  , system::error_code ec4 = {})
   {
      BOOST_ASIO_CORO_REENTER (coro_) for (;;)
      {
         BOOST_ASIO_CORO_YIELD
         conn_->resv_.async_resolve(asio::prepend(std::move(self), order_t {}));

         logger_.on_resolve(ec0, conn_->resv_.results());

         if (ec0) {
            self.complete(ec0);
            return;
         }

         BOOST_ASIO_CORO_YIELD
         conn_->ctor_.async_connect(
            conn_->next_layer().next_layer(),
            conn_->resv_.results(),
            asio::prepend(std::move(self), order_t {}));

         logger_.on_connect(ec0, conn_->ctor_.endpoint());

         if (ec0) {
            self.complete(ec0);
            return;
         }

         if (conn_->use_ssl()) {
            BOOST_ASIO_CORO_YIELD
            conn_->next_layer().async_handshake(
               asio::ssl::stream_base::client,
               asio::prepend(
                  asio::cancel_after(
                     runner_->cfg_.ssl_handshake_timeout,
                     std::move(self)
                  ),
                  order_t {}
               )
            );

            logger_.on_ssl_handshake(ec0);

            if (ec0) {
               self.complete(ec0);
               return;
            }
         }

         conn_->reset();

         // Note: Order is important here because the writer might
         // trigger an async_write before the async_hello thereby
         // causing an authentication problem.
         BOOST_ASIO_CORO_YIELD
         asio::experimental::make_parallel_group(
            [this](auto token) { return runner_->async_hello(*conn_, logger_, token); },
            [this](auto token) { return conn_->health_checker_.async_ping(*conn_, logger_, token); },
            [this](auto token) { return conn_->health_checker_.async_check_timeout(*conn_, logger_, token);},
            [this](auto token) { return conn_->reader(logger_, token);},
            [this](auto token) { return conn_->writer(logger_, token);}
         ).async_wait(
            asio::experimental::wait_for_one_error(),
            std::move(self));

         if (order[0] == 0 && !!ec0) {
            self.complete(ec0);
            return;
         }

         if (order[0] == 2 && ec2 == error::pong_timeout) {
            self.complete(ec1);
            return;
         }

         // The receive operation must be cancelled because channel
         // subscription does not survive a reconnection but requires
         // re-subscription.
         conn_->cancel(operation::receive);

         if (!conn_->will_reconnect()) {
            conn_->cancel(operation::reconnection);
            self.complete(ec3);
            return;
         }

         // It is safe to use the writer timer here because we are not
         // connected.
         conn_->writer_timer_.expires_after(conn_->cfg_.reconnect_wait_interval);

         BOOST_ASIO_CORO_YIELD
         conn_->writer_timer_.async_wait(asio::prepend(std::move(self), order_t {}));
         if (ec0) {
            self.complete(ec0);
            return;
         }

         if (!conn_->will_reconnect()) {
            self.complete(asio::error::operation_aborted);
            return;
         }

         conn_->reset_stream();
      }
   }
};

template <class Executor>
class runner {
public:
   runner(Executor ex, config cfg)
   : cfg_{cfg}
   { }

   void set_config(config const& cfg)
   {
      cfg_ = cfg;
   }

   template <class Connection, class Logger, class CompletionToken>
   auto async_run(Connection& conn, Logger l, CompletionToken token)
   {
      return asio::async_compose
         < CompletionToken
         , void(system::error_code)
         >(runner_op<runner, Connection, Logger>{this, &conn, l}, token, conn);
   }

private:

   template <class, class, class> friend class runner_op;
   template <class, class, class> friend struct hello_op;

   template <class Connection, class Logger, class CompletionToken>
   auto async_hello(Connection& conn, Logger l, CompletionToken token)
   {
      return asio::async_compose
         < CompletionToken
         , void(system::error_code)
         >(hello_op<runner, Connection, Logger>{this, &conn, l}, token, conn);
   }

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
