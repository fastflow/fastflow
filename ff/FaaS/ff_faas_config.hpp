#ifndef FF_FAAS_CONFIG_HPP
#define FF_FAAS_CONFIG_HPP

#include <string>
#include <unordered_map>
#include <fstream>
#include <sstream>
#include <vector>
#include <memory>
#include <stdexcept>
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
            if (!faas_ifs.is_open()) {
                throw std::runtime_error("Cannot open FaaS config file: " + faasFile);
            }

            {
                rapidjson::IStreamWrapper isw(faas_ifs);
                rapidjson::Document doc;
                doc.ParseStream(isw);
                if(doc.HasParseError()) 
                    throw std::runtime_error("Error parsing function config JSON: " + std::string(rapidjson::GetParseError_En(doc.GetParseError())));
                
                if (!doc.IsObject() || !doc.HasMember("faas_backends") || !doc["faas_backends"].IsArray()) {
                    throw std::runtime_error("Invalid or missing 'faas_backends' array in FaaS config");
                }

                for (const auto& backend : doc["faas_backends"].GetArray()) {
                    if (!backend.IsObject() || backend.MemberCount() != 3) {
                        throw std::runtime_error("Invalid backend entry, expected an object with 3 members");
                    }

                    if (!backend.HasMember("faas_name") || !backend["faas_name"].IsString() ||
                        !backend.HasMember("faas_type") || !backend["faas_type"].IsString() ||
                        !backend.HasMember("faas_config")) {
                        throw std::runtime_error("Invalid backend entry");
                    }

                    std::string name = backend["faas_name"].GetString();
                    if (faas_map.find(name) != faas_map.end()) {
                        throw std::runtime_error("Duplicate faas_name: " + name);
                    }

                    std::string type = backend["faas_type"].GetString();

                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    backend["faas_config"].Accept(writer);

                    faas_map.emplace(std::move(name), FaasEntry{ std::move(type), buffer.GetString() });
                }
            }

            // Parse function config
            std::ifstream function_ifs(functionFile);
            if (!function_ifs.is_open()) {
                throw std::runtime_error("Cannot open function config file: " + functionFile);
            }

            {
                rapidjson::IStreamWrapper isw(function_ifs);
                rapidjson::Document doc;
                doc.ParseStream(isw);

                if(doc.HasParseError()) 
                    throw std::runtime_error("Error parsing function config JSON: " + std::string(rapidjson::GetParseError_En(doc.GetParseError())));
                

                if (!doc.IsObject() || !doc.HasMember("faas_functions") || !doc["faas_functions"].IsArray()) {
                    throw std::runtime_error("Invalid or missing 'faas_functions' array in function config");
                }

                for (const auto& func : doc["faas_functions"].GetArray()) {
                    if (!func.IsObject() || func.MemberCount() != 3) {
                        throw std::runtime_error("Invalid function entry, expected an object with 3 members");
                    }
                    if (!func.HasMember("function_name") || !func["function_name"].IsString() ||
                        !func.HasMember("faas_name") || !func["faas_name"].IsString() ||
                        !func.HasMember("function_config")) {
                        throw std::runtime_error("Invalid function entry");
                    }

                    std::string fname = func["function_name"].GetString();
                    if (function_map.find(fname) != function_map.end()) {
                        throw std::runtime_error("Duplicate function_name: " + fname);
                    }

                    std::string faas_name = func["faas_name"].GetString();
                    if (faas_map.find(faas_name) == faas_map.end()) {
                        throw std::runtime_error("Function '" + fname + "' refers to undefined faas_name: " + faas_name);
                    }

                    rapidjson::StringBuffer buffer;
                    rapidjson::Writer<rapidjson::StringBuffer> writer(buffer);
                    func["function_config"].Accept(writer);

                    function_map.emplace(std::move(fname), FunctionEntry{ std::move(faas_name), buffer.GetString() });
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

    private:
        struct FaasEntry {
            std::string faas_type;
            std::string faas_config;
        };

        struct FunctionEntry {
            std::string faas_name;
            std::string function_config;
        };

        std::unordered_map<std::string, FaasEntry> faas_map;
        std::unordered_map<std::string, FunctionEntry> function_map;
    };
}

#endif // FF_FAAS_CONFIG_HPP