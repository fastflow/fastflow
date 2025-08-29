/* Copyright (c) 2018-2024 Marcelo Zimbres Silva (mzimbres@gmail.com)
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE.txt)
 */

#ifndef BOOST_REDIS_HEALTH_CHECKER_HPP
#define BOOST_REDIS_HEALTH_CHECKER_HPP

#include <boost/redis/request.hpp>
#include <boost/redis/response.hpp>
#include <boost/redis/operation.hpp>
#include <boost/redis/adapter/any_adapter.hpp>
#include <boost/redis/config.hpp>
#include <boost/redis/operation.hpp>
#include <boost/asio/steady_timer.hpp>
#include <boost/asio/compose.hpp>
#include <boost/asio/consign.hpp>
#include <boost/asio/coroutine.hpp>
#include <boost/asio/post.hpp>
#include <memory>
#include <chrono>

namespace boost::redis::detail {

template <class HealthChecker, class Connection, class Logger>
class ping_op {
public:
   HealthChecker* checker_ = nullptr;
   Connection* conn_ = nullptr;
   Logger logger_;
   asio::coroutine coro_{};

   template <class Self>
   void operator()(Self& self, system::error_code ec = {}, std::size_t = 0)
   {
      BOOST_ASIO_CORO_REENTER (coro_) for (;;)
      {
         if (checker_->ping_interval_ == std::chrono::seconds::zero()) {
            logger_.trace("ping_op (1): timeout disabled.");
            BOOST_ASIO_CORO_YIELD
            asio::post(std::move(self));
            self.complete({});
            return;
         }

         if (checker_->checker_has_exited_) {
            logger_.trace("ping_op (2): checker has exited.");
            self.complete({});
            return;
         }

         BOOST_ASIO_CORO_YIELD
         conn_->async_exec(checker_->req_, any_adapter(checker_->resp_), std::move(self));
         if (ec) {
            logger_.trace("ping_op (3)", ec);
            checker_->wait_timer_.cancel();
            self.complete(ec);
            return;
         }

         // Wait before pinging again.
         checker_->ping_timer_.expires_after(checker_->ping_interval_);

         BOOST_ASIO_CORO_YIELD
         checker_->ping_timer_.async_wait(std::move(self));
         if (ec) {
            logger_.trace("ping_op (4)", ec);
            self.complete(ec);
            return;
         }
      }
   }
};

template <class HealthChecker, class Connection, class Logger>
class check_timeout_op {
public:
   HealthChecker* checker_ = nullptr;
   Connection* conn_ = nullptr;
   Logger logger_;
   asio::coroutine coro_{};

   template <class Self>
   void operator()(Self& self, system::error_code ec = {})
   {
      BOOST_ASIO_CORO_REENTER (coro_) for (;;)
      {
         if (checker_->ping_interval_ == std::chrono::seconds::zero()) {
            logger_.trace("check_timeout_op (1): timeout disabled.");
            BOOST_ASIO_CORO_YIELD
            asio::post(std::move(self));
            self.complete({});
            return;
         }

         checker_->wait_timer_.expires_after(2 * checker_->ping_interval_);

         BOOST_ASIO_CORO_YIELD
         checker_->wait_timer_.async_wait(std::move(self));
         if (ec) {
            logger_.trace("check_timeout_op (2)", ec);
            self.complete(ec);
            return;
         }

         if (checker_->resp_.has_error()) {
            // TODO: Log the error.
            logger_.trace("check_timeout_op (3): Response error.");
            self.complete({});
            return;
         }

         if (checker_->resp_.value().empty()) {
            logger_.trace("check_timeout_op (4): pong timeout.");
            checker_->ping_timer_.cancel();
            conn_->cancel(operation::run);
            checker_->checker_has_exited_ = true;
            self.complete(error::pong_timeout);
            return;
         }

         if (checker_->resp_.has_value()) {
            checker_->resp_.value().clear();
         }
      }
   }
};

template <class Executor>
class health_checker {
private:
   using timer_type =
      asio::basic_waitable_timer<
         std::chrono::steady_clock,
         asio::wait_traits<std::chrono::steady_clock>,
         Executor>;

public:
   health_checker(Executor ex)
   : ping_timer_{ex}
   , wait_timer_{ex}
   {
      req_.push("PING", "Boost.Redis");
   }

   void set_config(config const& cfg)
   {
      req_.clear();
      req_.push("PING", cfg.health_check_id);
      ping_interval_ = cfg.health_check_interval;
   }

   void cancel()
   {
      ping_timer_.cancel();
      wait_timer_.cancel();
   }

   template <class Connection, class Logger, class CompletionToken>
   auto async_ping(Connection& conn, Logger l, CompletionToken token)
   {
      return asio::async_compose
         < CompletionToken
         , void(system::error_code)
         >(ping_op<health_checker, Connection, Logger>{this, &conn, l}, token, conn, ping_timer_);
   }

   template <class Connection, class Logger, class CompletionToken>
   auto async_check_timeout(Connection& conn, Logger l, CompletionToken token)
   {
      checker_has_exited_ = false;
      return asio::async_compose
         < CompletionToken
         , void(system::error_code)
         >(check_timeout_op<health_checker, Connection, Logger>{this, &conn, l}, token, conn, wait_timer_);
   }

private:
   template <class, class, class> friend class ping_op;
   template <class, class, class> friend class check_timeout_op;

   timer_type ping_timer_;
   timer_type wait_timer_;
   redis::request req_;
   redis::generic_response resp_;
   std::chrono::steady_clock::duration ping_interval_ = std::chrono::seconds{5};
   bool checker_has_exited_ = false;
};

} // boost::redis::detail

#endif // BOOST_REDIS_HEALTH_CHECKER_HPP
