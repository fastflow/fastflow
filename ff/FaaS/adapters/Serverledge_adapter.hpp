/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file connector.hpp
 *  \ingroup <TODO>
 *
 *  \brief  Defines the ff_faas_function_adapter class for a specific FAAS framework
 *
 *  It contains the definition of the \p ff_faas_function_adapter class,
 *  with features oriented offloading to FAAS systems.
 *
 */

/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/*
#ifndef SERVERLEDGE_ADAPTER_HPP
#define SERVERLEDGE_ADAPTER_HPP

#include <ff/ff_faas.hpp>
#include <ff/FaaS/ff_faas_function_adapter.hpp>
#include <ff/FaaS/ff_faas_function.hpp>
#include <simdutf/simdutf.h>
#include <simdutf/simdutf.cpp>
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/error/en.h>
#include <rapidjson/schema.h>

using namespace httplib;
using namespace std;
using namespace rapidjson;
using namespace simdutf;

class Serverledge_adapter: public ff_faas_function_adapter {
    public:

        Serverledge_adapter(ff::ff_faas_function_base& faas_fun, int, char**): ff_faas_function_adapter(faas_fun) {
            PRINT_DBG("Serverledge_adapter constructor called");
        }

        bool ff_faas_function_adapter_init() {
            PRINT_DBG("Initializing Serverledge_adapter...");
            return true;
        }

        bool ff_faas_function_adapter_run() {
            PRINT_DBG("Running Serverledge_adapter...");
            return launch_http_server();
        }

    private:

        void handleHTTPRequest(const Request& req, Response& res) {
            PRINT_DBG("Handling HTTP request: " << req.body);
            Document requestJson;
            ParseResult parseResult = requestJson.Parse(req.body.c_str());
            if (!parseResult) {
                res.status = 500;
                res.set_content(
                    "[Serverledge_adapter] Internal error: JSON parse error " + string(GetParseError_En(parseResult.Code())) + 
                    " at offset " + to_string(parseResult.Offset()), 
                    "text/plain"
                );
                PRINT_DBG("Error parsing JSON: " << GetParseError_En(parseResult.Code()) << " at offset " << parseResult.Offset());
                return;
            }

            if (!requestJson.HasMember("Params")) {
                res.status = 400;
                res.set_content("[Serverledge_adapter] Parameters error: parameter \"Params\" missed!\n", "text/plain");
                PRINT_DBG("Missing 'Params' field in JSON request.");
                return;
            }
            else
                if (!requestJson["Params"].IsObject()) {
                    res.status = 400;
                    res.set_content("[Serverledge_adapter] Parameters error: parameter \"Params\" is not a valid Json object!\n", "text/plain");
                    PRINT_DBG("Field 'Params' is not a valid JSON object.");
                    return;
                }

            const Value& params = requestJson["Params"];
            if (!params.HasMember("p")) {
                res.status = 400;
                res.set_content("[Serverledge_adapter] Parameters error: parameter \"p\" missed!\n", "text/plain");
                PRINT_DBG("Missing parameter 'p' in 'Params'.");
                return;
            }
            else
                if (!params["p"].IsString()) {
                    res.status = 400;
                    res.set_content("[Serverledge_adapter] Parameters error: parameter \"p\" is not a valid Json string!\n", "text/plain");
                    PRINT_DBG("Parameter 'p' is not a valid JSON string.");
                    return;
                }

            const Value& p = requestJson["Params"]["p"];
            size_t rSize = p.GetStringLength();
            size_t maxLength = maximal_binary_length_from_base64(p.GetString(), rSize);
            PRINT_DBG("Decoding Base64, input size: " << rSize << ", maxLength: " << maxLength);

            char* resultData = new char[maxLength];
            result Res = base64_to_binary(p.GetString(), rSize, resultData, simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
            if(Res.error) {
                res.status = 500;
                res.set_content("[Serverledge_adapter] Internal error: conversion from Base64 to binary for the input parameters failed", "text/plain");
                PRINT_DBG("Base64 to binary conversion failed.");
                return;
            }

            unique_ptr<dataBuffer> inputBuffer = make_unique<dataBuffer>();
            inputBuffer->setBuffer(resultData, Res.count, true);

            PRINT_DBG("Calling faas_fun.handle_request...");
            unique_ptr<dataBuffer> outputBuf = faas_fun.handle_request(move(inputBuffer));
            if (!outputBuf) {
                res.status = 500;
                res.set_content("[Serverledge_adapter] Internal error: failed to handle request", "text/plain");
                PRINT_DBG("faas_fun.handle_request failed.");
                return;
            }

            size_t json_response_middle_size = simdutf::base64_length_from_binary(outputBuf->getLen(), simdutf::base64_options::base64_default);
            std::unique_ptr<char[]> json_response_middle = std::make_unique<char[]>(json_response_middle_size);

            auto result = simdutf::binary_to_base64(outputBuf->getPtr(), outputBuf->getLen(), json_response_middle.get());
            if (result != json_response_middle_size) {
                res.status = 500;
                res.set_content("[Serverledge_adapter] Internal error: conversion from binary to Base64 for the output parameters failed. Expected size: " + std::to_string(json_response_middle_size) + ", got: " + std::to_string(result), "text/plain");
                PRINT_DBG("Base64 conversion for output failed.");
                return;
            }

            size_t total_len = json_response_starting_size + json_response_middle_size + json_response_ending_size;

            res.set_content_provider(
            total_len,
            "application/json",
            [&](size_t offset, size_t length, DataSink& sink) {
                size_t written = 0;

                if (offset < json_response_starting_size) {
                    size_t part = std::min(length, json_response_starting_size - offset);
                    sink.write(json_response_starting + offset, part);
                    written += part;
                }

                if (written < length && offset + written < json_response_starting_size + json_response_middle_size) {
                    size_t middle_offset = offset > json_response_starting_size ? offset - json_response_starting_size : 0;
                    size_t part = std::min(length - written, json_response_middle_size - middle_offset);
                    sink.write(json_response_middle.get() + middle_offset, part);
                    written += part;
                }

                if (written < length && offset + written < total_len) {
                    size_t suffix_offset = offset > (json_response_starting_size + json_response_middle_size) ? offset - json_response_starting_size - json_response_middle_size : 0;
                    size_t part = std::min(length - written, json_response_ending_size - suffix_offset);
                    sink.write(json_response_ending + suffix_offset, part);
                    written += part;
                }

                return true; // tutto bene
            });

            PRINT_DBG("Request processed and response sent.");
        }

        bool launch_http_server() {
            PRINT_DBG("Launching HTTP server...");
            srv.new_task_queue = [] { return new ThreadPool(1,1); };
            srv.Post(entrypoint, [this](const Request& req, Response& res) {
                this->handleHTTPRequest(req, res);
            });

            srv.listen(IP, PORT);
            PRINT_DBG("Server listening on " << IP << ":" << PORT);
            return true;
        }

        httplib::Server srv;

        static inline constexpr char json_response_starting[] = R"({"success":true,"result":"{\"r\":\")";
        static inline constexpr char json_response_ending[] = R"(\"}","output":""})";
        static inline constexpr size_t json_response_starting_size = 35; // Length of the starting JSON response string
        static inline constexpr size_t json_response_ending_size = 17; // Length of the "returnOutput_y" string

        static inline constexpr char entrypoint[] = "/invoke";
        static inline constexpr char IP[] = "0.0.0.0";
        static inline constexpr int PORT = 8080;
};
*/

#ifndef SERVERLEDGE_ADAPTER_HPP
#define SERVERLEDGE_ADAPTER_HPP

#include <ff/ff_faas.hpp>
#include <ff/FaaS/ff_faas_function_adapter.hpp>
#include <ff/FaaS/ff_faas_function.hpp>
#include <simdutf/simdutf.h>
#include <simdutf/simdutf.cpp>
#include <boost/beast.hpp>
#include <boost/asio.hpp>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/error/en.h>

using namespace boost;
using namespace std;
using namespace rapidjson;
using namespace simdutf;

class Serverledge_adapter : public ff_faas_function_adapter {
public:
    Serverledge_adapter(ff::ff_faas_function_base& faas_fun, int, char**)
        : ff_faas_function_adapter(faas_fun), io_context_(), tcp_acceptor_(io_context_, asio::ip::tcp::endpoint(asio::ip::make_address(IP), PORT)) {
        PRINT_DBG("Serverledge_adapter constructor called");
    }

    bool ff_faas_function_adapter_init() {
        PRINT_DBG("Initializing Serverledge_adapter...");
        return true;
    }

    bool ff_faas_function_adapter_run() {
        PRINT_DBG("Running Serverledge_adapter...");
        return launch_http_server();
    }

private:
    void handleHTTPRequest(asio::ip::tcp::socket& socket) {
        try {
            beast::flat_buffer buffer;  // Use flat_buffer for efficient storage of incoming data.
            beast::http::request<beast::http::string_body> req;
            beast::http::read(socket, buffer, req);  // Read the request into the buffer

            if (req.method() != beast::http::verb::post || req.target() != entrypoint) {
               sendErrorResponse(socket, beast::http::status::not_found,
                "[Serverledge_adapter] Invalid endpoint or method. Use POST on /invoke");
                PRINT_DBG("Invalid endpoint or method. Received: " << req.method_string() << " " << req.target());
                return;
            }

            PRINT_DBG("Handling HTTP request: " << req.body());

            // Parse the JSON request
            Document requestJson;
            ParseResult parseResult = requestJson.Parse(req.body().c_str());
            if (!parseResult) {
                sendErrorResponse(socket, beast::http::status::internal_server_error, "[Serverledge_adapter] Internal error: JSON parse error " + string(GetParseError_En(parseResult.Code())) + 
                    " at offset " + to_string(parseResult.Offset()));
                PRINT_DBG("Error parsing JSON: " << GetParseError_En(parseResult.Code()) << " at offset " << parseResult.Offset());
                return;
            }

            if (!requestJson.HasMember("Params")) {
                sendErrorResponse(socket, beast::http::status::bad_request, "[Serverledge_adapter] Parameters error: parameter \"Params\" missed!\n");
                PRINT_DBG("Missing 'Params' field in JSON request.");
                return;
            }
            else if (!requestJson["Params"].IsObject()) {
                sendErrorResponse(socket, beast::http::status::bad_request, "[Serverledge_adapter] Parameters error: parameter \"Params\" is not a valid Json object!\n");
                PRINT_DBG("Field 'Params' is not a valid JSON object.");
                return;
            }

            const Value& params = requestJson["Params"];
            if (!params.HasMember("p")) {
                sendErrorResponse(socket, beast::http::status::bad_request, "[Serverledge_adapter] Parameters error: parameter \"p\" missed!\n");
                PRINT_DBG("Missing parameter 'p' in 'Params'.");
                return;
            }
            else if (!params["p"].IsString()) {
                sendErrorResponse(socket, beast::http::status::bad_request, "[Serverledge_adapter] Parameters error: parameter \"p\" is not a valid Json string!\n");
                PRINT_DBG("Parameter 'p' is not a valid JSON string.");
                return;
            }

            const Value& p = requestJson["Params"]["p"];
            size_t rSize = p.GetStringLength();
            size_t maxLength = maximal_binary_length_from_base64(p.GetString(), rSize);
            PRINT_DBG("Decoding Base64, input size: " << rSize << ", maxLength: " << maxLength);

            char* resultData = new char[maxLength];
            result Res = base64_to_binary(p.GetString(), rSize, resultData, simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
            if(Res.error) {
                sendErrorResponse(socket, beast::http::status::internal_server_error, "[Serverledge_adapter] Internal error: conversion from Base64 to binary for the input parameters failed");
                PRINT_DBG("Base64 to binary conversion failed.");
                return;
            }

            unique_ptr<dataBuffer> inputBuffer = make_unique<dataBuffer>();
            inputBuffer->setBuffer(resultData, Res.count, true);

            PRINT_DBG("Calling faas_fun.handle_request...");
            unique_ptr<dataBuffer> outputBuf = faas_fun.handle_request(std::move(inputBuffer));
            if (!outputBuf) {
                sendErrorResponse(socket, beast::http::status::internal_server_error, "[Serverledge_adapter] Internal error: failed to handle request");
                PRINT_DBG("faas_fun.handle_request failed.");
                return;
            }

            size_t json_response_middle_size = simdutf::base64_length_from_binary(outputBuf->getLen(), simdutf::base64_options::base64_default);
            std::unique_ptr<char[]> json_response_middle = std::make_unique<char[]>(json_response_middle_size);

            auto result = simdutf::binary_to_base64(outputBuf->getPtr(), outputBuf->getLen(), json_response_middle.get());
            if (result != json_response_middle_size) {
                sendErrorResponse(socket, beast::http::status::internal_server_error, "[Serverledge_adapter] Internal error: conversion from binary to Base64 for the output parameters failed.");
                PRINT_DBG("Base64 conversion for output failed.");
                return;
            }

            PRINT_DBG("JSON da inviare: " 
                << string(json_response_starting, json_response_starting_size) 
                << string(json_response_middle.get(), json_response_middle_size) 
                << string(json_response_ending, json_response_ending_size));

            // Create HTTP response
            beast::http::response<beast::http::string_body> res{beast::http::status::ok, req.version()};
            res.set(beast::http::field::content_type, "application/json");
            beast::http::write(socket, res); 

            // Efficiently write to the response body without copying multiple times
            asio::write(socket, asio::buffer(json_response_starting, json_response_starting_size));
            asio::write(socket, asio::buffer(json_response_middle.get(), json_response_middle_size));
            asio::write(socket, asio::buffer(json_response_ending, json_response_ending_size));
            socket.close();
        } catch (const std::exception& e) {
            sendErrorResponse(socket, beast::http::status::internal_server_error, "[Serverledge_adapter] Internal error: " + string(e.what()));
            PRINT_DBG("Exception: " << e.what());
        }
    }

    void sendErrorResponse(asio::ip::tcp::socket& socket, beast::http::status statusCode, const string& message) {
        beast::http::response<beast::http::string_body> res{statusCode, 11};
        res.set(beast::http::field::content_type, "text/plain");
        res.body() = message;
        res.content_length(message.size());
        beast::http::write(socket, res);
    }

    bool launch_http_server() {
        try {
            PRINT_DBG("Launching HTTP server...");
            asio::ip::tcp::socket socket(io_context_);
            while(true) {
                tcp_acceptor_.accept(socket);
                PRINT_DBG("Accepted connection.");
                handleHTTPRequest(socket);    
            }
            return true;
        } catch (const std::exception& e) {
            PRINT_DBG("Server error: " << e.what());
            return false;
        }
    }

    asio::io_context io_context_;
    asio::ip::tcp::acceptor tcp_acceptor_;

    static inline constexpr char json_response_starting[] = R"({"success":true,"result":"{\"r\":\")";
    static inline constexpr char json_response_ending[] = R"(\"}","output":""})";
    static inline constexpr size_t json_response_starting_size = 35; // Length of the starting JSON response string
    static inline constexpr size_t json_response_ending_size = 17; // Length of the "returnOutput_y" string

    static inline constexpr char IP[] = "0.0.0.0";
    static inline constexpr int PORT = 8080;
    static inline constexpr char entrypoint[] = "/invoke";
};

#endif /* SERVERLEDGE_ADAPTER_HPP */