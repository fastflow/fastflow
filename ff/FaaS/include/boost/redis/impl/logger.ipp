/* Copyright (c) 2018-2024 Marcelo Zimbres Silva (mzimbres@gmail.com)
 *
 * Distributed under the Boost Software License, Version 1.0. (See
 * accompanying file LICENSE.txt)
 */

#include <boost/redis/logger.hpp>
#include <boost/system/error_code.hpp>
#include <iostream>
#include <iterator>

namespace boost::redis
{

void logger::write_prefix()
{
   if (!std::empty(prefix_))
      std::clog << prefix_;
}

void logger::on_resolve(system::error_code const& ec, asio::ip::tcp::resolver::results_type const& res)
{
   if (level_ < level::info)
      return;

   write_prefix();

   std::clog << "resolve results: ";

   if (ec) {
      std::clog << ec.message() << std::endl;
   } else {
      auto begin = std::cbegin(res);
      auto end = std::cend(res);

      if (begin == end)
         return;

      std::clog << begin->endpoint();
      for (auto iter = std::next(begin); iter != end; ++iter)
         std::clog << ", " << iter->endpoint();
   }

   std::clog << std::endl;
}

void logger::on_connect(system::error_code const& ec, asio::ip::tcp::endpoint const& ep)
{
   if (level_ < level::info)
      return;

   write_prefix();

   std::clog << "connected to ";

   if (ec)
      std::clog << ec.message() << std::endl;
   else
      std::clog << ep;

   std::clog << std::endl;
}

void logger::on_ssl_handshake(system::error_code const& ec)
{
   if (level_ < level::info)
      return;

   write_prefix();

   std::clog << "SSL handshake: " << ec.message() << std::endl;
}

void
logger::on_write(
   system::error_code const& ec,
   std::string const& payload)
{
   if (level_ < level::info)
      return;

   write_prefix();

   if (ec)
      std::clog << "writer_op: " << ec.message();
   else
      std::clog << "writer_op: " << std::size(payload) << " bytes written.";

   std::clog << std::endl;
}

void logger::on_read(system::error_code const& ec, std::size_t n)
{
   if (level_ < level::info)
      return;

   write_prefix();

   if (ec)
      std::clog << "reader_op: " << ec.message();
   else
      std::clog << "reader_op: " << n << " bytes read.";

   std::clog << std::endl;
}

void
logger::on_hello(
   system::error_code const& ec,
   generic_response const& resp)
{
   if (level_ < level::info)
      return;

   write_prefix();

   if (ec) {
      std::clog << "hello_op: " << ec.message();
      if (resp.has_error())
         std::clog << " (" << resp.error().diagnostic << ")";
   } else {
      std::clog << "hello_op: Success";
   }

   std::clog << std::endl;
}

void logger::trace(std::string_view message)
{
   if (level_ < level::debug)
      return;

   write_prefix();

   std::clog << message << std::endl;
}

void logger::trace(std::string_view op, system::error_code const& ec)
{
   if (level_ < level::debug)
      return;

   write_prefix();

   std::clog << op << ": " << ec.message() << std::endl;
}

} // boost::redis
