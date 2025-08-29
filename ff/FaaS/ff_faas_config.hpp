/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file connector.hpp
 *  \ingroup ff
 *
 *  \brief  Defines the ff_faas_config class to parse the configuration JSON files for FAAS frameworks
 *
 *  It contains the definition of the \p ff_faas_config class,
 *  to parse the two configuration files to define parameters for FAAS functions and for FAAS frameworks
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

#ifndef FF_FAAS_CONFIG_HPP
#define FF_FAAS_CONFIG_HPP

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
#include <ff/ff_faas_configuration.hpp>
#include <rapidjson/document.h>
#include <rapidjson/istreamwrapper.h>
#include <rapidjson/stringbuffer.h>
#include <rapidjson/writer.h>
#include <rapidjson/error/en.h>


namespace ff {

    class ff_faas_config {
    public:
        explicit ff_faas_config(const std::string& faasFile, const std::string& functionFile) {
            // Parse FaaS config
            std::ifstream faas_ifs(faasFile);
            if (!faas_ifs.is_open()) 
                throw std::runtime_error("Cannot open FaaS config file: " + faasFile);            
            {
                rapidjson::IStreamWrapper isw(faas_ifs);
                rapidjson::Document doc;
                doc.ParseStream(isw);

                if (doc.IsNull()) 
                    throw std::runtime_error("Empty parsing FaaS backend config JSON file: " + faasFile);
                
                if(doc.HasParseError()) 
                    throw std::runtime_error("Error parsing FaaS backend config JSON file: " + std::string(rapidjson::GetParseError_En(doc.GetParseError())));
                
                if (!doc.IsObject() || !doc.HasMember("faas_backends") || !doc["faas_backends"].IsArray()) 
                    throw std::runtime_error("Invalid or missing 'faas_backends' array in FaaS config");

                for (const auto& backend : doc["faas_backends"].GetArray()) {
                    if (!backend.IsObject() || backend.MemberCount() != 3) 
                        throw std::runtime_error("Invalid backend entry, expected an object with 3 members");

                    if (!backend.HasMember("faas_name") || !backend["faas_name"].IsString() ||
                        !backend.HasMember("faas_type") || !backend["faas_type"].IsString() ||
                        !backend.HasMember("faas_config") || !backend["faas_config"].IsObject()) 
                        throw std::runtime_error("Invalid backend entry");
                    
                    std::string name = backend["faas_name"].GetString();
                    if (faas_map.find(name) != faas_map.end()) 
                        throw std::runtime_error("Duplicate faas_name: " + name);

                    std::string type = backend["faas_type"].GetString();

                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    backend["faas_config"].Accept(writer);

                    faas_map.emplace(std::move(name), FaasEntry{ std::move(type), buffer.GetString() });
                }
            }

            // Parse function config
            std::ifstream function_ifs(functionFile);
            if (!function_ifs.is_open())
                throw std::runtime_error("Cannot open function config file: " + functionFile);            

            {
                rapidjson::IStreamWrapper isw(function_ifs);
                rapidjson::Document doc;
                doc.ParseStream(isw);
                if (doc.IsNull()) 
                    throw std::runtime_error("Empty parsing function config JSON file: " + functionFile);
                
                if(doc.HasParseError()) 
                    throw std::runtime_error("Error parsing function config JSON file: " + std::string(rapidjson::GetParseError_En(doc.GetParseError())));
                
                if (!doc.IsObject() || !doc.HasMember("faas_functions") || !doc["faas_functions"].IsArray()) 
                    throw std::runtime_error("Invalid or missing 'faas_functions' array in function config");                

                for (const auto& func : doc["faas_functions"].GetArray()) {
                    if (!func.IsObject() || func.MemberCount() < 3) 
                        throw std::runtime_error("Invalid function entry, expected an object with at least 3 members");
                    
                    if (!func.HasMember("function_name") || !func["function_name"].IsString() ||
                        !func.HasMember("faas_name") || !func["faas_name"].IsString() ||
                        !func.HasMember("function_config") || !func["function_config"].IsObject()) 
                        throw std::runtime_error("Invalid function entry");
                    
                    std::string fname = func["function_name"].GetString();

                    if (func.HasMember("initial_parallelism")) 
                        if (!func["initial_parallelism"].IsUint() || func["initial_parallelism"].GetUint() < 1) 
                            throw std::runtime_error("Invalid initial_parallelism for function: " + std::string(func["function_name"].GetString()));
                                            
                    unsigned long initial_parallelism = func.HasMember("initial_parallelism") ? func["initial_parallelism"].GetUint() : DEFAULT_PARALLELISM_DEGREE;

                    if(func.HasMember("stats_collection")) 
                        if(!func["stats_collection"].IsBool()) 
                            throw std::runtime_error("Invalid stats_collection for function: " + std::string(func["function_name"].GetString()));                        
                    
                    bool stats_collection = func.HasMember("stats_collection") ? func["stats_collection"].GetBool() : false;

                    if (function_map.find(fname) != function_map.end()) 
                        throw std::runtime_error("Duplicate function_name: " + fname);
                    
                    std::string faas_name = func["faas_name"].GetString();
                    if (faas_map.find(faas_name) == faas_map.end()) 
                        throw std::runtime_error("Function '" + fname + "' refers to undefined faas_name: " + faas_name);
                    
                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    func["function_config"].Accept(writer);

                    function_map.emplace(std::move(fname), FunctionEntry{ std::move(faas_name), buffer.GetString(), initial_parallelism, stats_collection});
                }
            }
        }

        bool hasFaas(const std::string& faasName) const {
            return faas_map.find(faasName) != faas_map.end();
        }

        bool hasFunction(const std::shared_ptr<std::string> functionName) const {
            return function_map.find(*functionName) != function_map.end();
        }

        const std::string& getFaasType(const std::string& faasName) const {
            return faas_map.at(faasName).faas_type;
        }

        const std::string& getFaasConfig(const std::string& faasName) const {
            return faas_map.at(faasName).faas_config;
        }

        const std::string& getFunctionFaasName(const std::shared_ptr<std::string> functionName) const {
            return function_map.at(*functionName).faas_name;
        }

        const std::string& getFunctionFaasType(const std::shared_ptr<std::string> functionName) const {
            const std::string& faasName = function_map.at(*functionName).faas_name;
            return faas_map.at(faasName).faas_type;
        }

        const std::string& getFunctionFaasConfig(const std::shared_ptr<std::string> functionName) const {
            const std::string& faasName = function_map.at(*functionName).faas_name;
            return faas_map.at(faasName).faas_config;
        }

        const std::string& getFunctionConfig(const std::shared_ptr<std::string> functionName) const {
            return function_map.at(*functionName).function_config;
        }

        unsigned long getFunctionInitialParallelism(const std::shared_ptr<std::string> functionName) const {
            return function_map.at(*functionName).initial_parallelism;
        }

        bool isStatsCollectionEnabled(const std::shared_ptr<std::string> functionName) const {
            return function_map.at(*functionName).stats_collection;
        }

    private:
        struct FaasEntry {
            std::string faas_type;
            std::string faas_config;
        };

        struct FunctionEntry {
            std::string faas_name;
            std::string function_config;
            unsigned long initial_parallelism;
            bool stats_collection;
        };

        std::unordered_map<std::string, FaasEntry> faas_map;
        std::unordered_map<std::string, FunctionEntry> function_map;
    };
}

#endif // FF_FAAS_CONFIG_HPP