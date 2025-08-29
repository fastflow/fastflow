/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file connector.hpp
 *  \ingroup <TODO>
 *
 *  \brief  Defines the Serverledge_connector class to connect to the Serverledge FaaS framework
 *
 *  It contains the definition of the \p Serverledge_connector class,
 *  with features oriented offloading to Serverledge FaaS systems.
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

#ifndef SERVERLEDGE_CONNECTOR_HPP
#define SERVERLEDGE_CONNECTOR_HPP

#include <ff/FaaS/ff_faas_connector.hpp>
#include <ff/FaaS/ff_faas_config.hpp>
#include <ff/ff_faas.hpp>
#include <simdutf/simdutf.h>
#include <simdutf/simdutf.cpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/error/en.h>
#include <rapidjson/schema.h>
#include <httplib.h>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <mutex>

class Serverledge_connector : public ff_faas_connector {
public:
    Serverledge_connector(const std::shared_ptr<const ff::ff_faas_config> faasConfig,
                          const std::shared_ptr<std::string> functionName)
        : ff_faas_connector(faasConfig, functionName) {
            
        PRINT_DBG("Constructor called for function: " + *functionName);

        std::call_once(initSchemasFlag, &Serverledge_connector::preprocessSchemas);
    }

    ~Serverledge_connector() override {
        PRINT_DBG("Destructor called.");
    }

    std::unique_ptr<dataBuffer> invokeFaasFunction(std::unique_ptr<dataBuffer> payload) override {
        if (!payload) {
            std::cerr << "[Serverledge_connector] Function invocation error: payload is nullptr for function: " << *functionName << std::endl;
            return nullptr;
        }

        PRINT_DBG("InvokeFaasFunction called for: " + *functionName);

        char* payloadBuff = payload->getPtr();
        size_t payloadSize = payload->getLen();

        size_t sendBufferSize = simdutf::base64_length_from_binary(payloadSize, simdutf::base64_options::base64_default);
        
        std::unique_ptr<char[]> sendBuffer = std::make_unique<char[]>(sendBufferSize);

        auto result = simdutf::binary_to_base64(payloadBuff, payloadSize, sendBuffer.get());
        if (result != sendBufferSize) {
            std::cerr << "[Serverledge_connector] Function invocation error: Base64 encoding failed. Expected size: " << sendBufferSize << ", got: " << result << " for function: " << *functionName << std::endl;
            return nullptr;
        }

        size_t total_size = sendBufferSize + json_payload_starting_size + json_payload_ending_size;

        auto& req_it = Serverledge_connector::function_registration_map.at(*functionName);

        std::shared_ptr<rapidjson::Document> req_json_doc = req_it.first;
        std::string EntryPoint = "/invoke/" + std::string(req_json_doc->GetObject()["Name"].GetString());
        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli)
            cli = std::make_unique<httplib::Client>(host, port);

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES; 
        int attempt = 0;
        while (attempt < maxRetries) {
            httplib::Result HTTPres = cli->Post(EntryPoint, headers,
                    total_size,
                    [&](size_t offset, size_t length, httplib::DataSink &sink) {
                        if (offset < json_payload_starting_size) {
                            size_t len = std::min(length, json_payload_starting_size - offset);
                            sink.write(json_payload_starting + offset, len);
                        } else if (offset < json_payload_starting_size + sendBufferSize) {
                            size_t relative = offset - json_payload_starting_size;
                            size_t len = std::min(length, sendBufferSize - relative);
                            sink.write(sendBuffer.get() + relative, len);
                        } else if (offset < total_size) {
                            size_t relative = offset - json_payload_starting_size - sendBufferSize;
                            size_t len = std::min(length, json_payload_ending_size - relative);
                            sink.write(json_payload_ending + relative, len);
                        } else {
                            return false; // offset out of range
                        }
                        return true;
                    },
                    "application/json");
            if (!HTTPres) {
                std::cerr << "[Serverledge_connector] Invocation request for function " << *functionName << " failed: " << httplib::to_string(HTTPres.error()) << std::endl;
                return nullptr;
            }

            switch (HTTPres->status) {
                case 200: {
                    
                    std::vector<char> respBuffer(HTTPres->body.begin(), HTTPres->body.end());
                    PRINT_DBG("[Serverledge_connector] Risposta HTTP ricevuta: " << respBuffer.data());

                    std::shared_ptr<rapidjson::Document> res_json_doc = std::make_shared<rapidjson::Document>();
                    if (res_json_doc->Parse(HTTPres->body.c_str()).HasParseError()) {
                        std::cerr << "[Serverledge_connector] Function invocation error: JSON parse error at offset "
                                  << res_json_doc->GetErrorOffset() << ": "
                                  << rapidjson::GetParseError_En(res_json_doc->GetParseError())
                                  << " for function " << *functionName << std::endl;
                        return nullptr;
                    }

                    if (!validateJsonAgainstSchema(res_json_doc, *HTTPInvocationResponseSchema)) {
                        std::cerr << "[Serverledge_connector] Function invocation error: JSON validation failed for function " << *functionName << std::endl;
                        return nullptr;   
                    }

                    if (!res_json_doc->GetObject()["Success"].GetBool()) {
                        std::cerr << "[Serverledge_connector] Function invocation error: Serverledge execution error for function " << *functionName << std::endl;
                        return nullptr;  
                    }

                    const rapidjson::Value& result = res_json_doc->GetObject()["Result"];
                    res_json_doc->Parse(result.GetString(), result.GetStringLength()); 
                    if (res_json_doc->HasParseError()) {
                        std::cerr << "[Serverledge_connector] Function invocation error: JSON validation failed for function " << *functionName << std::endl;
                        return nullptr;
                    }

                    if(!res_json_doc->HasMember("r") || !res_json_doc->GetObject()["r"].IsString()) {
                        std::cerr << "Errore: la risposta non contiene il campo 'r' o non è del tipo stringa\n";
                        return nullptr;
                    }

                    rapidjson::Value& r =  res_json_doc->GetObject()["r"];

                    size_t rSize = r.GetStringLength();
                    size_t maxLength = simdutf::maximal_binary_length_from_base64(r.GetString(), rSize);
                    char* resultData = new char[maxLength];
                    simdutf::result res = simdutf::base64_to_binary(r.GetString(), rSize, resultData, simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
                    if(res.error) {
                        std::cerr << "[Serverledge_connector] Function invocation error: conversion from Base64 to binary for the return parameters failed for function " << *functionName << std::endl;
                        return nullptr;    
                    }      
                    std::unique_ptr<dataBuffer> resultBuffer = std::make_unique<dataBuffer>();
                    resultBuffer->setBuffer(resultData, res.count, true);
                    
                    std::cout << "[Serverledge_connector] Function invoked successfully." << std::endl;
                    return resultBuffer; 
                }
                case 404:
                    std::cerr << "[Serverledge_connector] Function invocation error: Unknown function. " << std::endl;
                    return nullptr;
                case 429:
                    std::cerr << "[Serverledge_connector] Function invocation error: excessive load. " << std::endl;
                    return nullptr;
                case 500:
                    std::cerr << "[Serverledge_connector] Function invocation error: server internal error. Retrying..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(SERVERUNAVAILABLEDELAY));
                    attempt++;
                    break;
                default:
                    std::cerr << "[Serverledge_connector] Function invocation error: Unexpected error. " << std::endl;
                    return nullptr;
            }
        }
        std::cerr << "[Serverledge_connector] Function invocation error: Max retries reached for invocation of function " << *functionName << std::endl;
        return nullptr;
    }

    RegistrationResult registerFaasFunction() override {

        PRINT_DBG("registerFaasFunction called for: " + *functionName);
        
        std::unique_lock<std::mutex> lock(registrationMutex);
        const std::string& faasName = faasConfig->getFunctionFaasName(functionName);
        auto itFaasConfig = function_config_map.find(faasName);
        std::shared_ptr<rapidjson::Document> req_config_json_doc = std::make_shared<rapidjson::Document>();

        if (itFaasConfig == function_config_map.end()) {
            std::string faasConfigStr = faasConfig->getFaasConfig(faasName);  // Otteniamo la configurazione Faas dal faasConfig

            if (req_config_json_doc->Parse(faasConfigStr.c_str()).HasParseError()) {
                std::cerr << "[Serverledge_connector] Invalid JSON for FaasConfig: Parse error at offset "
                        << req_config_json_doc->GetErrorOffset() << ": "
                        << rapidjson::GetParseError_En(req_config_json_doc->GetParseError()) << std::endl;
                return REGISTRATION_ERROR;
            }

            // Schema validation
            if (!validateJsonAgainstSchema(req_config_json_doc, *faasConfigEntrySchema)) {
                std::cerr << "[Serverledge_connector] JSON validation failed for FaasConfig: " << faasName << std::endl;
                return REGISTRATION_ERROR;         
            }

            if(!req_config_json_doc->HasMember("port")) 
                req_config_json_doc->AddMember("port", rapidjson::Value(DEFAULT_PORT), req_config_json_doc->GetAllocator());               
        }

        std::shared_ptr<rapidjson::Document> req_json_doc = std::make_shared<rapidjson::Document>();
        auto it = function_registration_map.find(*functionName);
        if (it == function_registration_map.end()) {
            std::string functionConfig = faasConfig->getFunctionConfig(functionName);  
            if (req_json_doc->Parse(functionConfig.c_str()).HasParseError()) {
                std::cerr << "[Serverledge_connector] Invalid JSON: Parse error at offset "
                        << req_json_doc->GetErrorOffset() << ": "
                        << rapidjson::GetParseError_En(req_json_doc->GetParseError()) << std::endl;
                return REGISTRATION_ERROR;
            }

            if (!validateJsonAgainstSchema(req_json_doc, *functionConfigEntrySchema)) {
                std::cerr << "[Serverledge_connector] JSON validation failed for function: " << *functionName << std::endl;
                return REGISTRATION_ERROR;            
            }
                
            if (strcmp(req_json_doc->GetObject()["Runtime"].GetString(),RUNTIME)!=0) {
                if(req_json_doc->HasMember("CustomImage")) {
                    std::cerr << "[Serverledge_connector] Error: CustomImage is NOT required for non-custom runtimes." << std::endl;
                    return REGISTRATION_ERROR;
                }
                if(!req_json_doc->HasMember("Handler"))  {
                    std::cerr << "[Serverledge_connector] Error: Handler is required for non-custom runtimes." << std::endl;
                    return REGISTRATION_ERROR;  
                }

                if(!req_json_doc->HasMember("TarFunctionCode")) {
                    std::cerr << "[Serverledge_connector] Error: TarFunctionCode is required for non-custom runtimes." << std::endl;
                    return REGISTRATION_ERROR;  
                }
            }
            else {
                if(!req_json_doc->HasMember("CustomImage")) {
                    std::cerr << "[Serverledge_connector] Error: CustomImage is required for custom runtimes." << std::endl;
                    return REGISTRATION_ERROR;  
                }
            }                    

            if(sendHttpRegistrationRequest(req_json_doc,req_config_json_doc) == REGISTRATION_ERROR)
                return REGISTRATION_ERROR;            

            function_config_map[faasName] = req_config_json_doc;
            function_registration_map[*functionName] = std::make_pair(req_json_doc, 1);
            PRINT_DBG("Saved registration for function: " << *functionName);           
            lock.unlock(); 
            sendHTTPPrewarmingRequest();
            PRINT_DBG("Registration for function: " << *functionName << " finished succesfully.");
            return REGISTRATION_OK;
        }

        it->second.second++;

        lock.unlock(); 
        sendHTTPPrewarmingRequest();

        PRINT_DBG("Registration for function: " << *functionName << " finished: yet registered.");
        return YET_REGISTERED;
    }

    DeRegistrationResult deregisterFaasFunction() override {
        PRINT_DBG("deregisterFaasFunction called for: " + *functionName);

        std::lock_guard<std::mutex> lock(registrationMutex);

        auto it = function_registration_map.find(*functionName);
        if (it == function_registration_map.end()) {
            std::cerr << "[Serverledge_connector] Warning: trying to deregister non-existing function: " << *functionName << std::endl;
            return DEREGISTRATION_ERROR;
        }

        it->second.second--;
        PRINT_DBG("Remained registrations for function: " << *functionName << ": " << it->second.second);

        if (it->second.second <= 0) {
            PRINT_DBG("Last deregistration for function: " << *functionName << ". Cleaning.");
            if(sendHttpDeRegistrationRequest() == DEREGISTRATION_ERROR) 
                return DEREGISTRATION_ERROR;    

            function_registration_map.erase(it);
            return DEREGISTRATION_OK;
        }

        return NOT_YET_DEREGISTERED;
    }

private:

    DeRegistrationResult sendHttpDeRegistrationRequest() {
        auto& req_it = Serverledge_connector::function_registration_map.at(*functionName);

        std::shared_ptr<rapidjson::Document> req_json_doc = req_it.first;
        std::string jsonPayload = "{\"Name\": \"" + std::string(req_json_doc->GetObject()["Name"].GetString()) + "\"}";

        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli) 
            cli = std::make_unique<httplib::Client>(host, port);

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES; 
        int attempt = 0;
        while (attempt < maxRetries) {
            auto res = cli->Post("/delete", headers, jsonPayload, "application/json");

            if (!res) {
                std::cerr << "[HTTP] Deregistration request for function " << *functionName << " failed: " << httplib::to_string(res.error()) << std::endl;
                return DEREGISTRATION_ERROR;
            }

            switch (res->status) {
                case 200:
                    std::cout << "[Serverledge_connector] Function deregistered successfully: " << res->body << std::endl;
                    return DEREGISTRATION_OK;
                case 404:
                    std::cerr << "[Serverledge_connector] Function deregistration error: Unknown function. " << res->body << std::endl;
                    return NOT_REGISTERED;
                case 503:
                    std::cerr << "[Serverledge_connector] Function deregistration error: server unavailable. Retrying..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(SERVERUNAVAILABLEDELAY));
                    attempt++;
                    break;
                default:
                    std::cerr << "[Serverledge_connector] Function deregistration error: Unexpected error. " << res->body << std::endl;
                    return DEREGISTRATION_ERROR;
            }
        }
        std::cerr << "[Serverledge_connector] Function deregistration error: Max retries reached for deregistration of function " << *functionName << std::endl;
        return DEREGISTRATION_ERROR;
    }

    RegistrationResult sendHttpRegistrationRequest(std::shared_ptr<rapidjson::Document> req_json_doc, std::shared_ptr<rapidjson::Document> req_config_json_doc) {

        rapidjson::StringBuffer buffer;
        rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
        req_json_doc->Accept(writer);

        std::string jsonPayload = buffer.GetString();

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli)
            cli = std::make_unique<httplib::Client>(host, port);

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES;
        int attempt = 0;
        while (attempt < maxRetries) {
            auto res = cli->Post("/create", headers, jsonPayload, "application/json");

            if (!res) {
                std::cerr << "[Serverledge_connector] Registration request for function " << *functionName << " failed: " << httplib::to_string(res.error()) << std::endl;
                return REGISTRATION_ERROR;
            }

            switch (res->status) {
                case 200:
                    std::cout << "[Serverledge_connector] Function registered successfully: " << res->body << std::endl;
                    return REGISTRATION_OK;
                case 404:
                    std::cerr << "[Serverledge_connector] Function registration error: Invalid runtime. " << res->body << std::endl;
                    return REGISTRATION_ERROR;
                case 409:
                    std::cerr << "[Serverledge_connector] Function registration error:: Function already exists. " << res->body << std::endl;
                    return YET_REGISTERED;
                case 503:
                    std::cerr << "[Serverledge_connector] Function registration error:: server unavailable. Retrying..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(SERVERUNAVAILABLEDELAY)); 
                    attempt++;
                    break;
                default:
                    std::cerr << "[Serverledge_connector] Function registration error: Unexpected error. " << res->body << std::endl;
                    return REGISTRATION_ERROR;
            }
        }
        std::cerr << "[Serverledge_connector] Function registration error: Max retries reached for registration of function " << *functionName << std::endl;
        return REGISTRATION_ERROR;     
    }

    PrewarmingResult sendHTTPPrewarmingRequest() {

        PRINT_DBG("sendHTTPPrewarmingRequest called for: " + *functionName);
        std::shared_ptr<rapidjson::Document> req_json_doc = Serverledge_connector::function_registration_map.at(*functionName).first;

        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));

        std::string jsonPayload = "{\"Function\": \"" + std::string(req_json_doc->GetObject()["Name"].GetString()) + "\",\"Instances\": 1}";

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli)
            cli = std::make_unique<httplib::Client>(host, port);

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES;
        int attempt = 0;
        while (attempt < maxRetries) {
            auto res = cli->Post("/prewarm", headers, jsonPayload, "application/json");

            if (!res) {
                std::cerr << "[Serverledge_connector] Prewarming request for function " << *functionName << " failed: " << httplib::to_string(res.error()) << std::endl;
                return PREWARMING_ERROR;
            }

            switch (res->status) {
                case 200:
                    std::cout << "[Serverledge_connector] Function prewarmed successfully: " << res->body << std::endl;
                    return PREWARMING_OK;
                case 404:
                    std::cerr << "[Serverledge_connector] Prewarming function error: Unknown function. " << res->body << std::endl;
                    return PREWARMING_ERROR;
                case 503:
                    std::cerr << "[Serverledge_connector] Prewarming function error: server unavailable. Retrying..." << std::endl;
                    std::this_thread::sleep_for(std::chrono::seconds(SERVERUNAVAILABLEDELAY)); 
                    attempt++;
                    break;
                default:
                    std::cerr << "[Serverledge_connector] Prewarming function error: Unexpected error. " << res->body << std::endl;
                    return PREWARMING_ERROR;
            }
        }
        std::cerr << "[Serverledge_connector] Prewarming function error: Max retries reached for prewarming of function " << *functionName << std::endl;
        return PREWARMING_ERROR;         
    }

    static void preprocessSchemas() {
        rapidjson::Document schemaDoc;

        // Preprocessing schema FaasConfig
        schemaDoc.Parse(FaasConfigEntry_schemaStr);
        if (schemaDoc.HasParseError())
            throw std::runtime_error("[Serverledge_connector] FaasConfig schema parse error!\n");
            
        faasConfigEntrySchema = std::make_shared<rapidjson::SchemaDocument>(schemaDoc); 

        // Preprocessing schema FunctionConfig
        schemaDoc.Parse(FunctionConfigEntry_schemaStr);
        if (schemaDoc.HasParseError()) 
            throw std::runtime_error("[Serverledge_connector] FunctionConfig schema parse error!\n");
            
        functionConfigEntrySchema = std::make_shared<rapidjson::SchemaDocument>(schemaDoc);

        // Preprocessing schema HTTPInvocationResponse
        schemaDoc.Parse(HTTPInvocationResponse_schemaStr);
         if (schemaDoc.HasParseError()) 
            throw std::runtime_error("[Serverledge_connector] HTTPInvocationResponse schema parse error!\n");
            
        HTTPInvocationResponseSchema = std::make_shared<rapidjson::SchemaDocument>(schemaDoc); 

        PRINT_DBG("Schemas preprocessed and stored.");
    }

    bool validateJsonAgainstSchema(std::shared_ptr<rapidjson::Document> doc, rapidjson::SchemaDocument& schemaDoc) {
        rapidjson::SchemaValidator validator(schemaDoc);
        if (!doc->Accept(validator)) {
            rapidjson::StringBuffer sb;
            validator.GetInvalidSchemaPointer().StringifyUriFragment(sb);
            std::cerr << "[Serverledge_connector] Schema validation failed at: " << sb.GetString() << std::endl;

            sb.Clear();
            validator.GetInvalidDocumentPointer().StringifyUriFragment(sb);
            std::cerr << "[Serverledge_connector] Document error at: " << sb.GetString() << std::endl;
            return false;
        }
        return true;
    }

    static inline constexpr char FunctionConfigEntry_schemaStr[] = R"({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "Name":            { "type": "string" },
            "Runtime":         { "type": "string" },
            "MemoryMB":        { "type": "integer" },
            "CPUDemand":       { "type": "number" },
            "Handler":         { "type": "string" },
            "TarFunctionCode": { "type": "string" },
            "CustomImage":     { "type": "string" }
        },
        "required": ["Name","Runtime","MemoryMB"],
        "additionalProperties": false
    })";


    static inline constexpr char HTTPInvocationResponse_schemaStr[] = R"({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "Success": { "type": "boolean" },
            "Result": { "type": "string" },
            "ResponseTime": { "type": "number" },
            "IsWarmStart": { "type": "boolean" },
            "InitTime": { "type": "number" },
            "OffloadLatency": { "type": "number" },
            "Duration": { "type": "number" },
            "SchedAction": { "type": "string" }
        },
        "required": ["Success", "Result", "ResponseTime", "OffloadLatency", "Duration", "SchedAction"]
    })";

    static inline constexpr char FaasConfigEntry_schemaStr[] = R"({
        "$schema": "http://json-schema.org/draft-07/schema#",
        "type": "object",
        "properties": {
            "host": { "type": "string" },
            "port": { "type": "integer" }
        },
        "required": ["host"],
        "additionalProperties": false
    })";

    std::unique_ptr<httplib::Client> cli = nullptr;

    static inline constexpr char json_payload_starting[] = "{\"Params\":{\"p\":\"";
    static inline constexpr char json_payload_ending[] = "\"}}";
    static inline constexpr size_t json_payload_starting_size = 16; // Length of the starting JSON payload string
    static inline constexpr size_t json_payload_ending_size = 3; // Length of the ending

    alignas(CACHE_LINE_SIZE) static inline std::unordered_map<std::string, std::pair<std::shared_ptr<rapidjson::Document>, int>> function_registration_map;
    alignas(CACHE_LINE_SIZE) static inline std::unordered_map<std::string, std::shared_ptr<rapidjson::Document>> function_config_map;
    alignas(CACHE_LINE_SIZE) static inline std::mutex registrationMutex;

    alignas(CACHE_LINE_SIZE) static inline std::shared_ptr<rapidjson::SchemaDocument> HTTPInvocationResponseSchema;
    alignas(CACHE_LINE_SIZE) static inline std::shared_ptr<rapidjson::SchemaDocument> faasConfigEntrySchema;
    alignas(CACHE_LINE_SIZE) static inline std::shared_ptr<rapidjson::SchemaDocument> functionConfigEntrySchema;

    alignas(CACHE_LINE_SIZE) static inline std::once_flag initSchemasFlag;

    static inline constexpr char RUNTIME[] = "custom";
    static inline constexpr time_t CONN_TIMEOUT_SEC = 5; 
    static inline constexpr time_t CONN_TIMEOUT_USEC = 0;
    static inline constexpr time_t WRITE_TIMEOUT_SEC = 5; 
    static inline constexpr time_t WRITE_TIMEOUT_USEC = 0;
    static inline constexpr short int MAXRETRIES = 5;
    static inline constexpr int64_t SERVERUNAVAILABLEDELAY = 2.0;
    static inline constexpr int DEFAULT_PORT = 1323;

};

REGISTER_CONNECTOR("Serverledge", Serverledge_connector)

#endif // SERVERLEDGE_CONNECTOR_HPP