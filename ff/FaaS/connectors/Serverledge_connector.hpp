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

#include <ff/ff_faas.hpp>
#include <ff/FaaS/ff_faas_connector.hpp>
#include <chrono>
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
#include <shared_mutex>

#ifdef DEBUG
std::mutex debug_output_mutex;
#endif

class Serverledge_connector : public ff_faas_connector {
public:
    Serverledge_connector(const std::shared_ptr<const ff::ff_faas_config> faasConfig,
                          const std::shared_ptr<std::string> functionName)
        : ff_faas_connector(faasConfig, functionName), stats_collection(false) {
            
        PRINT_DBG(std::string("Constructor called for function: ") + *functionName);
        stats_collection = faasConfig->isStatsCollectionEnabled(functionName);
        std::call_once(initSchemasFlag, &Serverledge_connector::preprocessSchemas);
        PRINT_DBG(std::string("Constructor finished for function: ") + *functionName);
    }

    ~Serverledge_connector() override {
        PRINT_DBG(std::string("Destructor called."));
    }

    std::unique_ptr<dataBuffer> invokeFaasFunction(std::unique_ptr<dataBuffer> payload, std::shared_ptr<stats_entry>& stats) override {
        if (!payload) {
            std::cerr << "[Serverledge_connector] Function invocation error: payload is nullptr for function: " << *functionName << std::endl;
            return nullptr;
        }

        PRINT_DBG(std::string("InvokeFaasFunction called for: ") + *functionName);

        char* payloadBuff = payload->getPtr();
        size_t payloadSize = payload->getLen();

        size_t sendBufferSize = simdutf::base64_length_from_binary(payloadSize, simdutf::base64_options::base64_default);
        std::unique_ptr<char[]> sendBuffer = nullptr;
        try {
             sendBuffer = std::make_unique<char[]>(sendBufferSize);
        } catch (const std::bad_alloc& e) {
            std::cerr << "[Serverledge_connector] Function invocation error: Memory allocation failed for base64 encoding buffer for function: " << *functionName << ". Exception: " << e.what() << std::endl;            
            return nullptr;
        }

        auto result = simdutf::binary_to_base64(payloadBuff, payloadSize, sendBuffer.get());
        if (result != sendBufferSize) {
            std::cerr << "[Serverledge_connector] Function invocation error: Base64 encoding failed. Expected size: " << sendBufferSize << ", got: " << result << " for function: " << *functionName << std::endl;
            return nullptr;
        }
        size_t total_size = 0;
        size_t json_payload_starting_size = 0;
        const char* json_payload_starting = nullptr;
        if(stats_collection) {
            init_stats(stats); 
            stats->Msg_dim_sent = sendBufferSize + json_payload_starting_stats_size + json_payload_ending_size;  
            json_payload_starting_size = json_payload_starting_stats_size;              
            json_payload_starting = json_payload_starting_stats;
        }
        else 
        {
            json_payload_starting_size = json_payload_starting_s;              
            json_payload_starting = json_payload_st;            
        }

        total_size = sendBufferSize + json_payload_starting_size + json_payload_ending_size;
        
        std::shared_lock<std::shared_mutex> lock(registrationMutex, std::defer_lock); // non blocca subito
        lock.lock(); 
         auto& req_it = Serverledge_connector::function_registration_map.at(*functionName);
        std::shared_ptr<rapidjson::Document> req_json_doc = req_it.first;
        std::string EntryPoint = "/invoke/" + std::string(req_json_doc->GetObject()["Name"].GetString());
        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli)
            try {
                cli = std::make_unique<httplib::Client>(host, port);
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Function invocation error: Memory allocation failed to create HTTP client for function " << *functionName << ". Exception: " << e.what() << std::endl;
                return nullptr;
            }
        lock.unlock();

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES; 
        int attempt = 0;
        while (attempt < maxRetries) {
            httplib::Result HTTPres;
            try{
                std::chrono::high_resolution_clock::time_point T_call_start;
                if(stats_collection) 
                    T_call_start = std::chrono::high_resolution_clock::now();
                
                PRINT_DBG(std::string("[Serverledge_connector] Function ") + *functionName + std::string(" invoked with message: ") + std::string(json_payload_starting) + std::string(sendBuffer.get()) + std::string(json_payload_ending) + std::string("\n"));

                HTTPres = cli->Post(EntryPoint, headers,
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
                if(stats_collection) {                  
                    std::chrono::duration<double> T_call_dur = std::chrono::high_resolution_clock::now() - T_call_start;
                    T_faas_call = std::chrono::duration<double, std::micro>(T_call_dur).count();                
                }
                    
                if (!HTTPres) {
                    std::cerr << "[Serverledge_connector] Invocation request for function " << *functionName << " failed: " << httplib::to_string(HTTPres.error()) << std::endl;
                    return nullptr;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Serverledge_connector] Invocation request for function " << *functionName << " failed: Exception during HTTP POST. Exception: " << e.what() << std::endl;
                return nullptr;
            }

            switch (HTTPres->status) {
                case 200: {
                    PRINT_DBG(std::string("[Serverledge_connector] Function ") + *functionName + std::string(" returned with message: ") + HTTPres->body);
                    std::shared_ptr<rapidjson::Document> res_json_doc = nullptr;
                    try {                                      
                        res_json_doc = std::make_shared<rapidjson::Document>();
                    }
                    catch (const std::bad_alloc& e) {
                        std::cerr << "[Serverledge_connector] Function invocation error: Memory allocation failed for response JSON document for function " << *functionName << ". Exception: " << e.what() << std::endl;                        
                        return nullptr;
                    }

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

                    if(stats_collection) { 
                        if(res_json_doc->HasMember("IsWarmStart")) 
                            stats->is_warm = res_json_doc->GetObject()["IsWarmStart"].GetBool();                                         
                        stats->T_init_container = res_json_doc->GetObject()["InitTime"].GetDouble() * 1000000.0;                                                                                                 
                        if(res_json_doc->HasMember("OffloadLatency")) 
                             stats->T_offload = res_json_doc->GetObject()["OffloadLatency"].GetDouble() * 1000000.0;
                        // TODO: check if OffloadLatency is really included in ResponseTime
                        stats->T_faas_overhead = res_json_doc->GetObject()["ResponseTime"].GetDouble() * 1000000.0 + stats->T_offload * 1000000.0;
                    }

                    const rapidjson::Value& result = res_json_doc->GetObject()["Result"];
                    res_json_doc->Parse(result.GetString(), result.GetStringLength()); 
                    if (res_json_doc->HasParseError()) {
                        std::cerr << "[Serverledge_connector] Function invocation error: JSON validation failed for function " << *functionName << std::endl;
                        return nullptr;
                    }

                    if(!res_json_doc->HasMember("r") || !res_json_doc->GetObject()["r"].IsString()) {
                        std::cerr << "[Serverledge_connector] Function invocation error: answer does not contain an 'r' field or it is not a string\n";
                        return nullptr;
                    }

                    if(stats_collection) {
                        stats->Msg_dim_recv = HTTPres->body.size();

                        if(!res_json_doc->HasMember("s") || !res_json_doc->GetObject()["s"].IsDouble()) {
                            std::cerr << "[Serverledge_connector] Function invocation error for statistics collection: answer does not contain an 's' field or it is not a double\n";
                            return nullptr;
                        }
                        else {
                            stats->T_fun_exec = res_json_doc->GetObject()["s"].GetDouble();
                            stats->T_faas_overhead -= stats->T_fun_exec;
                            stats->T_comm = T_faas_call - stats->T_fun_exec - stats->T_faas_overhead; 
                        }
                    }

                    rapidjson::Value& r =  res_json_doc->GetObject()["r"];

                    size_t rSize = r.GetStringLength();
                    size_t maxLength = simdutf::maximal_binary_length_from_base64(r.GetString(), rSize);
                    std::unique_ptr<char[]> resultData = std::make_unique<char[]>(maxLength);
                    simdutf::result res = simdutf::base64_to_binary(r.GetString(), rSize, resultData.get(), simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
                    if(res.error) {
                        std::cerr << "[Serverledge_connector] Function invocation error: conversion from Base64 to binary for the return parameters failed for function " << *functionName << std::endl;
                        return nullptr;    
                    }      
                    std::unique_ptr<dataBuffer> resultBuffer = std::make_unique<dataBuffer>();
                    resultBuffer->setBuffer(resultData.release(), res.count, true);
                    
                    PRINT_DBG("Function invoked successfully.");
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

    RegistrationResult registerFaasFunction(double& T_reg) override {

        T_reg = 0;
        PRINT_DBG(std::string("registerFaasFunction called for: ") + *functionName);  

        std::unique_lock<std::shared_mutex> lock(registrationMutex);

        std::chrono::high_resolution_clock::time_point T_reg_start;
        if(stats_collection) 
            T_reg_start = std::chrono::high_resolution_clock::now();

        // I recover the backend installation from the configuration backend file 
        const std::string& faasName = faasConfig->getFunctionFaasName(functionName);
        // I search if I have yet registered the installation before in the configuration backend map
        auto itFaasConfig = function_config_map.find(faasName);
        std::shared_ptr<rapidjson::Document> req_config_json_doc = nullptr;

        // I register the new backend installation, if not present in the configuration backend map
        if (itFaasConfig == function_config_map.end()) {

            std::string faasConfigStr = faasConfig->getFaasConfig(faasName); 
            // I create and fill a new backend installation json document for future requests
            try {
                req_config_json_doc = std::make_shared<rapidjson::Document>();
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Registration error: Memory allocation failed for FaasConfig JSON document for function: " << *functionName << ". Exception: " << e.what() << std::endl;
                return REGISTRATION_ERROR;
            }

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
        else 
            req_config_json_doc = itFaasConfig->second;

        std::shared_ptr<rapidjson::Document> req_json_doc = nullptr;

        auto it = function_registration_map.find(*functionName);

        // I register the new function on the backend installation, if not present in the function_registration map
        if (it == function_registration_map.end()) {

            std::string functionConfig = faasConfig->getFunctionConfig(functionName);  

            try {
                req_json_doc = std::make_shared<rapidjson::Document>();
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Registration error: Memory allocation failed for request JSON document for function: " << *functionName << ". Exception: " << e.what() << std::endl;
                return REGISTRATION_ERROR;
            }

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
            try {
                if (itFaasConfig == function_config_map.end())
                    function_config_map[faasName] = req_config_json_doc;
                function_registration_map[*functionName] = std::make_pair(req_json_doc, 1);
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Registration error: Memory allocation failed while saving registration for function: " << *functionName << ". Exception: " << e.what() << std::endl;
                return REGISTRATION_ERROR;
            }

            PRINT_DBG(std::string("Saved registration for function: ") + *functionName);           
            lock.unlock(); 

            PRINT_DBG(std::string("Registration for function: ") + *functionName + std::string(" finished succesfully."));   
            if(stats_collection) {                  
                std::chrono::duration<double> T_reg_dur = std::chrono::high_resolution_clock::now() - T_reg_start;
                T_reg = std::chrono::duration<double, std::micro>(T_reg_dur).count();                
            }         
            return REGISTRATION_OK;
        }

        it->second.second++;

        lock.unlock(); 

        PRINT_DBG(std::string("Registration for function: ") + *functionName + std::string(" finished: yet registered."));
        
        if(stats_collection) {                  
            std::chrono::duration<double> T_reg_dur = std::chrono::high_resolution_clock::now() - T_reg_start;
            T_reg = std::chrono::duration<double, std::micro>(T_reg_dur).count();                
        }  

        return YET_REGISTERED;
    }

    DeRegistrationResult deregisterFaasFunction(double& T_dereg) override {

        T_dereg = 0;
        PRINT_DBG(std::string("deregisterFaasFunction called for: ") + *functionName);

        std::chrono::high_resolution_clock::time_point T_dereg_start;
        if(stats_collection) 
            T_dereg_start = std::chrono::high_resolution_clock::now();

        std::lock_guard<std::shared_mutex> lock(registrationMutex);

        auto it = function_registration_map.find(*functionName);
        if (it == function_registration_map.end()) {
            std::cerr << "[Serverledge_connector] Warning: trying to deregister non-existing function: " << *functionName << std::endl;
            return DEREGISTRATION_ERROR;
        }

        it->second.second--;
        PRINT_DBG(std::string("Remained registrations for function: ") + *functionName + std::string(": ") + std::to_string(it->second.second));

        if (it->second.second <= 0) {
            PRINT_DBG(std::string("Last deregistration for function: ") + *functionName + std::string(". Cleaning."));
            if(sendHttpDeRegistrationRequest() == DEREGISTRATION_ERROR) 
                return DEREGISTRATION_ERROR;    

            function_registration_map.erase(it);
            if(stats_collection) {                  
                std::chrono::duration<double> T_dereg_dur = std::chrono::high_resolution_clock::now() - T_dereg_start;
                T_dereg = std::chrono::duration<double, std::micro>(T_dereg_dur).count();                
            }  
            return DEREGISTRATION_OK;
        }

        if(stats_collection) {                  
            std::chrono::duration<double> T_dereg_dur = std::chrono::high_resolution_clock::now() - T_dereg_start;
            T_dereg = std::chrono::duration<double, std::micro>(T_dereg_dur).count();                
        }

        return NOT_YET_DEREGISTERED;
    }

    PrewarmingResult prewarmingFaasFunction(unsigned long num, double& T_prewarm) {
        T_prewarm = 0;
        PRINT_DBG(std::string(" prewarmingFaasFunction called for: ") + *functionName);

        std::chrono::high_resolution_clock::time_point T_prewarm_start;
        if(stats_collection) 
            T_prewarm_start = std::chrono::high_resolution_clock::now();

        if(sendHTTPPrewarmingRequest(num) == PREWARMING_ERROR)
            return PREWARMING_ERROR;
        
        if(stats_collection) {                  
            std::chrono::duration<double> T_prewarm_dur = std::chrono::high_resolution_clock::now() - T_prewarm_start;
            T_prewarm = std::chrono::duration<double, std::micro>(T_prewarm_dur).count();                
        }
        return PREWARMING_OK;
    }

private:

    void init_stats(std::shared_ptr<stats_entry>& stats) {
        if(stats_collection) {
            stats = std::make_shared<stats_entry>();
            stats->is_warm = false;
            stats->T_comm = 0.0;
            stats->T_total = 0.0;
            stats->T_faas_overhead = 0.0;
            stats->T_ff_overhead = 0.0;
            stats->T_fun_exec = 0.0;
            stats->T_init_container = 0.0;
            stats->T_offload = 0.0;
            stats->Msg_dim_sent = 0;
            stats->Msg_dim_recv = 0;
        }        
    }

    DeRegistrationResult sendHttpDeRegistrationRequest() {
        auto& req_it = Serverledge_connector::function_registration_map.at(*functionName);

        std::shared_ptr<rapidjson::Document> req_json_doc = req_it.first;
        std::string jsonPayload;
        try {
            jsonPayload = "{\"Name\": \"" + std::string(req_json_doc->GetObject()["Name"].GetString()) + "\"}";
        } catch (const std::bad_alloc& e) {
            std::cerr << "[Serverledge_connector] Deregistration error: Memory allocation failed for JSON payload for function: " << *functionName << ". Exception: " << e.what() << std::endl;
            return DEREGISTRATION_ERROR;
        }

        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli) 
            try{
                cli = std::make_unique<httplib::Client>(host, port);
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Deregistration error: Memory allocation failed to create HTTP client for function " << *functionName << ". Exception: " << e.what() << std::endl;
                return DEREGISTRATION_ERROR;
            }

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
                    PRINT_DBG("Function deregistered successfully: " + res->body);
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
            httplib::Result res;
            try {
                PRINT_DBG(std::string("Msg sent for registration:") + jsonPayload);
                res = cli->Post("/create", headers, jsonPayload, "application/json");

                if (!res) {
                    std::cerr << "[Serverledge_connector] Registration request for function " << *functionName << " failed: " << httplib::to_string(res.error()) << std::endl;
                    return REGISTRATION_ERROR;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Serverledge_connector] Registration request for function " << *functionName << " failed: Exception during HTTP POST. Exception: " << e.what() << std::endl;
                return REGISTRATION_ERROR;
            }   

            switch (res->status) {
                case 200:
                    PRINT_DBG("Function registered successfully: " + res->body);
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

    PrewarmingResult sendHTTPPrewarmingRequest(unsigned long num) {

        PRINT_DBG(std::string("sendHTTPPrewarmingRequest called for: ") + *functionName);

        std::shared_lock<std::shared_mutex> lock(registrationMutex);
        std::shared_ptr<rapidjson::Document> req_json_doc = Serverledge_connector::function_registration_map.at(*functionName).first;

        auto& req_config_json_doc = function_config_map.at(faasConfig->getFunctionFaasName(functionName));
        std::string jsonPayload;
        try {
            jsonPayload = "{\"Function\": \"" + std::string(req_json_doc->GetObject()["Name"].GetString()) + "\",\"Instances\":" + std::to_string(num) + "}";
        } catch (const std::bad_alloc& e) {
            std::cerr << "[Serverledge_connector] Prewarming error: Memory allocation failed for JSON payload for function: " << *functionName << ". Exception: " << e.what() << std::endl;
            return PREWARMING_ERROR;
        }

        const std::string& host = req_config_json_doc->GetObject()["host"].GetString();
        int port = req_config_json_doc->GetObject()["port"].GetInt();

        if(!cli)
            try{
                cli = std::make_unique<httplib::Client>(host, port);
            } catch (const std::bad_alloc& e) {
                std::cerr << "[Serverledge_connector] Prewarming error: Memory allocation failed to create HTTP client for function " << *functionName << ". Exception: " << e.what() << std::endl;
                return PREWARMING_ERROR;
            }

        cli->set_connection_timeout(CONN_TIMEOUT_SEC, CONN_TIMEOUT_USEC); 
        cli->set_write_timeout(WRITE_TIMEOUT_SEC, WRITE_TIMEOUT_USEC); 

        httplib::Headers headers = {
            { "Content-Type", "application/json" }
        };

        int maxRetries = MAXRETRIES;
        int attempt = 0;
        while (attempt < maxRetries) {
            httplib::Result res;
            try {                 
                res = cli->Post("/prewarm", headers, jsonPayload, "application/json");
                if (!res) {
                   std::cerr << "[Serverledge_connector] Prewarming request for function " << *functionName << " failed: " << httplib::to_string(res.error()) << std::endl;
                    return PREWARMING_ERROR;
                }
            } catch (const std::exception& e) {
                std::cerr << "[Serverledge_connector] Prewarming request for function " << *functionName << " failed: Exception during HTTP POST. Exception: " << e.what() << std::endl;
                return PREWARMING_ERROR;
            }

            switch (res->status) {
                case 200:
                    PRINT_DBG("Function prewarmed successfully: " + res->body);
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

        PRINT_DBG(std::string("Schemas preprocessed and stored."));
    }

    bool inline validateJsonAgainstSchema(std::shared_ptr<rapidjson::Document> doc, rapidjson::SchemaDocument& schemaDoc) {
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
    bool stats_collection;
    double T_faas_call = 0.0;
    static inline constexpr char json_payload_st[] = "{\"Params\":{\"p\":\"";
    static inline constexpr char json_payload_starting_stats[] = "{\"Params\":{\"s\":\"\",\"p\":\"";
    static inline constexpr char json_payload_ending[] = "\"}}";
    static inline constexpr size_t json_payload_starting_s = 16; // Length of the starting JSON payload string
    static inline constexpr size_t json_payload_starting_stats_size = 23; // Length of the starting JSON payload string with stats
    static inline constexpr size_t json_payload_ending_size = 3; // Length of the ending

    alignas(CACHE_LINE_SIZE) static inline std::unordered_map<std::string, std::pair<std::shared_ptr<rapidjson::Document>, int>> function_registration_map;
    alignas(CACHE_LINE_SIZE) static inline std::unordered_map<std::string, std::shared_ptr<rapidjson::Document>> function_config_map;
    alignas(CACHE_LINE_SIZE) static inline std::shared_mutex registrationMutex;

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