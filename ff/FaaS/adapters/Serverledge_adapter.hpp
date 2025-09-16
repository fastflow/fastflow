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

#ifndef SERVERLEDGE_ADAPTER_HPP
#define SERVERLEDGE_ADAPTER_HPP

#include<ff/ff_faas.hpp>
#include <ff/FaaS/ff_faas_function_adapter.hpp>
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

#ifdef DEBUG
std::mutex debug_output_mutex;
#endif

class Serverledge_adapter : public ff_faas_function_adapter {
public:
    Serverledge_adapter(int, char**)
        : io_context(), tcp_acceptor(io_context, asio::ip::tcp::endpoint(asio::ip::make_address(IP), PORT)),
        socket(io_context) {
        PRINT_DBG("Serverledge_adapter constructor called");
    }

    bool ff_faas_function_adapter_init() {
        PRINT_DBG("Initializing Serverledge_adapter...");
        return true;
    }

    // it should return nullptr in case of an error
    std::unique_ptr<dataBuffer> ff_faas_function_adapter_getRequest(unique_ptr<std::string>& error_msg, bool& stats_collection) {
        try {
            tcp_acceptor.accept(socket);
            PRINT_DBG("Accepted connection.");            
            unique_ptr<dataBuffer> inputBuffer = handleHTTPRequest(error_msg); 
            stats_collection = this->stats_collection;             
            return inputBuffer;
        } catch (const std::exception& e) {
            error_msg = make_unique<std::string>("[Serverledge_adapter] Internal error: " + string(e.what()));
            PRINT_DBG(std::string("Server error: ") + string(e.what()));
            return nullptr;
        }
    }

    // it returns false in case of an error, true otherwise
    bool ff_faas_function_adapter_sendResponse(unique_ptr<dataBuffer> outputBuf, unique_ptr<string>& error_msg, double T_fun_exec) {
        try{            
            size_t json_response_middle_size = simdutf::base64_length_from_binary(outputBuf->getLen(), simdutf::base64_options::base64_default);
            std::unique_ptr<char[]> json_response_middle = std::make_unique<char[]>(json_response_middle_size);

            auto result = simdutf::binary_to_base64(outputBuf->getPtr(), outputBuf->getLen(), json_response_middle.get());
            if (result != json_response_middle_size) { 
                PRINT_DBG("Base64 conversion for output failed.");
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: conversion from binary to Base64 for the output parameters failed");
                return false;
            }

            if(stats_collection) {
                PRINT_DBG(string("JSON da inviare con statistiche: ") 
                    + string(json_response_starting, json_response_starting_size) 
                    + string(json_response_middle.get(), json_response_middle_size) 
                    + string(json_response_stats, json_response_stats_size) 
                    + std::to_string(T_fun_exec)
                    + string(json_response_stats_ending, json_response_stats_ending_size));
                }
                else {
                    PRINT_DBG(string("JSON da inviare: ")
                    + string(json_response_starting, json_response_starting_size) 
                    + string(json_response_middle.get(), json_response_middle_size) 
                    + string(json_response_ending, json_response_ending_size));
                }

            // Create HTTP response
            beast::http::response<beast::http::string_body> res{beast::http::status::ok, version};
            res.set(beast::http::field::content_type, "application/json");
            beast::http::write(socket, res); 

            // Efficiently write to the response body without copying multiple times
            asio::write(socket, asio::buffer(json_response_starting, json_response_starting_size));
            asio::write(socket, asio::buffer(json_response_middle.get(), json_response_middle_size));

            if(stats_collection) {
                asio::write(socket, asio::buffer(json_response_stats, json_response_stats_size));                
                asio::write(socket, asio::buffer(std::to_string(T_fun_exec)));
                asio::write(socket, asio::buffer(json_response_stats_ending,json_response_stats_ending_size));
            }   
            else
                asio::write(socket, asio::buffer(json_response_ending, json_response_ending_size)); 

            socket.close();            
        } catch (const std::bad_alloc& e) {
            error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: memory allocation failed");
            PRINT_DBG(string("Memory allocation failed for output parameters: ") + string(e.what()));
            return false;
        }
        catch (const boost::system::system_error& e) {
            error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal Network/system error: " + string(e.what()));
            PRINT_DBG(string("Network/system error: ") + string(e.what()));
            return false;
        } 
        catch (const std::exception& e) {
            error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: " + string(e.what()));
            PRINT_DBG(string("Exception: ") + string(e.what()));
            return false;
        } 
        return true;
    }

    bool ff_faas_function_adapter_sendErrorResponse(std::unique_ptr<std::string> message) {
        try {
            beast::http::response<beast::http::string_body> res{beast::http::status::ok, version};
            res.set(beast::http::field::content_type, "application/json");
            beast::http::write(socket, res);

            asio::write(socket, asio::buffer(json_response_starting_error, json_response_starting_error_size));
            asio::write(socket, asio::buffer(message->c_str(), message->size()));
            asio::write(socket, asio::buffer(json_response_ending, json_response_ending_size));

            socket.close();
        } 
        catch (const boost::system::system_error& e) {
            std::cerr << "[Serverledge_adapter] Network/system error: " << e.what() << std::endl;
            return false;
        }
        catch (const std::exception& e) {
            std::cerr << "[Serverledge_adapter] Unexpected exception: " << e.what() << std::endl;
            return false;
        }
        return true;
    }

private:
    unique_ptr<dataBuffer> handleHTTPRequest(std::unique_ptr<std::string>& error_msg) {
        try {
            beast::flat_buffer buffer;  // Use flat_buffer for efficient storage of incoming data.
            beast::http::request<beast::http::string_body> req;
            beast::http::read(socket, buffer, req);  // Read the request into the buffer

            if (req.method() != beast::http::verb::post || req.target() != entrypoint) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Invalid entrypoint or method. Use POST on /invoke");                
                PRINT_DBG(string("Invalid entrypoint or method. Received: ") + string(req.method_string()) + " " + string(req.target()));
                return nullptr;
            }
            version = req.version();
            PRINT_DBG(string("Handling HTTP request: ") + req.body());

            // Parse the JSON request
            Document requestJson;
            ParseResult parseResult = requestJson.Parse(req.body().c_str());
            if (!parseResult) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: JSON parse error " + string(GetParseError_En(parseResult.Code())) + 
                    " at offset " + to_string(parseResult.Offset()));
                PRINT_DBG(string("Error parsing JSON: ") + string(GetParseError_En(parseResult.Code())) + string(" at offset ") + to_string(parseResult.Offset()));
                return nullptr;
            }

            if (!requestJson.HasMember("Params")) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Parameters error: parameter \"Params\" missed!\n");
                PRINT_DBG("Missing 'Params' field in JSON request.");
                return nullptr;
            }
            else if (!requestJson["Params"].IsObject()) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Parameters error: parameter \"Params\" is not a valid Json object!\n");
                PRINT_DBG("Field 'Params' is not a valid JSON object.");
                return nullptr;
            }

            const Value& params = requestJson["Params"];
            if (!params.HasMember("p")) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Parameters error: parameter \"p\" missed!\n");
                PRINT_DBG("Missing parameter 'p' in 'Params'.");
                return nullptr;
            }
            else if (!params["p"].IsString()) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Parameters error: parameter \"p\" is not a valid Json string!\n");
                PRINT_DBG("Parameter 'p' is not a valid JSON string.");
                return nullptr;
            }

            if(params.HasMember("s")) {
                PRINT_DBG("Statistics collection requested by Serverledge_connector.");
                stats_collection = true;
            }

            const Value& p = requestJson["Params"]["p"];
            size_t rSize = p.GetStringLength();
            size_t maxLength = maximal_binary_length_from_base64(p.GetString(), rSize);
            PRINT_DBG(string("Decoding Base64, input size: ") + to_string(rSize) + string(", maxLength: ") + to_string( maxLength));

            std::unique_ptr<char[]> resultData;
            try {
                resultData = std::make_unique<char[]>(maxLength);
            } catch (const std::bad_alloc& e) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: memory allocation failed creating Base64 encoding buffer");
                PRINT_DBG(string("Memory allocation failed for input parameters: ") + string(e.what()));
                return nullptr;
            }

            result Res = base64_to_binary(p.GetString(), rSize, resultData.get(), simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
            if(Res.error) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: conversion from Base64 to binary for the input parameters failed");
                PRINT_DBG("Base64 to binary conversion failed.");
                return nullptr;
            }
            PRINT_DBG("Base64 to binary conversion of the request succeed.");

            std::unique_ptr<dataBuffer> inputBuffer;
            try {
                inputBuffer = std::make_unique<dataBuffer>();
            } catch (const std::bad_alloc& e) {
                error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: memory allocation failed creating input dataBuffer");
                PRINT_DBG(string("Memory allocation failed creating input dataBuffer: ") + string(e.what()));
                return nullptr;
            }

            inputBuffer->setBuffer(resultData.release(), Res.count, true);
            PRINT_DBG(string("Handling HTTP request: ") + req.body() + string(" finished."));
            return inputBuffer;

        } catch (const std::exception& e) {
            error_msg = std::make_unique<std::string>("[Serverledge_adapter] Internal error: " + string(e.what()));
            PRINT_DBG(string("Exception: ") + string(e.what()));
            return nullptr;
        }
    }

    unsigned int version;
    asio::io_context io_context;
    asio::ip::tcp::acceptor tcp_acceptor;
    asio::ip::tcp::socket socket;
    bool stats_collection = false;
    static inline constexpr char json_response_starting[] = R"({"success":true,"result":"{\"r\":\")";
    static inline constexpr char json_response_ending[] = R"(\"}","output":""})";
    static inline constexpr char json_response_stats_ending[] = R"(}","output":""})";
    static inline constexpr char json_response_stats[] = R"(\",\"s\":)";     
    static inline constexpr char json_response_starting_error[] = R"({"success":false,"result":"{\"r\":\"\")";
    static inline constexpr size_t json_response_starting_size = 35; 
    static inline constexpr size_t json_response_starting_error_size = 38; 
    static inline constexpr size_t json_response_stats_size = 9;
    static inline constexpr size_t json_response_ending_size = 17; 
    static inline constexpr size_t json_response_stats_ending_size = 15; 


    static inline constexpr char IP[] = "0.0.0.0";
    static inline constexpr int PORT = 8080;
    static inline constexpr char entrypoint[] = "/invoke";
};

#endif /* SERVERLEDGE_ADAPTER_HPP */