/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file connector.hpp
 *  \ingroup <TODO>
 *
 *  \brief  Defines the ff_faas_connector class for a specific FAAS framework
 *
 *  It contains the definition of the \p ff_faas_connector class,
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

#ifndef FF_FAAS_CONNECTOR_HPP
#define FF_FAAS_CONNECTOR_HPP

#include <ff/FaaS/ff_faas_config.hpp>
#include <ff/distributed/ff_network.hpp>
#include <atomic>
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <mutex>
#include <iostream>

//Stats for a call to the FaaS , in microseconds
// 0: not available or not applicable or really zero
struct stats_entry {
    double T_comm; // Communication time NO
    double T_total; // Total time FATTO
    double T_reg; // Registration time, if applicable  FATTO
    double T_dereg; // Deregistration time, if applicable FATTO 
    double T_prewarm; // Prewarming time, if applicable FATTO 
    double T_faas_overhead; // FaaS overhead time, T_init_container and T_offload included FATTO 
                            // TODO: I don't know if T_offload is included really for Serverledge
    double T_ff_overhead; // FastFlow overhead time FATTO
    double T_fun_exec; // Fun execution time on the FaaS FATTO
    double T_init_container; // Time for container initialization ( only if is_warm == false ) FATTO
    double T_offload; // Time for offloading the function to another node of the FaaS (0: no offloading) FATTO
    uint64_t Msg_dim_sent; // Message dimension sent, in bytes FATTO
    uint64_t Msg_dim_recv; // Message dimension received, in bytes FATTO
    bool is_warm; // True if the function is executed on a warm container, false otherwise FATTO
};


#define REGISTER_CONNECTOR(NAME, CLASS)                                                                 \
    namespace                                                                                           \
    {                                                                                                   \
        struct CLASS##Registrar                                                                         \
        {                                                                                               \
            CLASS##Registrar()                                                                          \
            {                                                                                           \
                ff_faas_connector_factory::instance().registerConnector(                                \
                    NAME,                                                                               \
                    [](std::shared_ptr<const ff::ff_faas_config> conf, const std::shared_ptr<std::string> functionName) -> std::unique_ptr<ff_faas_connector> { \
                        return std::make_unique<CLASS>(conf, functionName);                              \
                    });                                                                                 \
            }                                                                                           \
            ~CLASS##Registrar()                                                                         \
            {                                                                                           \
                ff_faas_connector_factory::instance().deregisterConnector(NAME);                        \
            }                                                                                           \
        };                                                                                              \
        static CLASS##Registrar global_##CLASS##_registrar;                                             \
    }

class ff_faas_connector
{
    public:
        enum RegistrationResult {
            REGISTRATION_OK,
            YET_REGISTERED,
            REGISTRATION_ERROR
        };

        enum DeRegistrationResult {
            DEREGISTRATION_OK,
            NOT_REGISTERED,
            NOT_YET_DEREGISTERED,
            DEREGISTRATION_ERROR
        };

        enum PrewarmingResult {
            PREWARMING_OK,
            PREWARMING_ERROR
        };

        virtual ~ff_faas_connector() = default;

        ff_faas_connector(const std::shared_ptr<const ff::ff_faas_config> faasConfig, const std::shared_ptr<std::string> functionName)
            : functionName(functionName) {
            if (!faasConfig) 
                throw std::invalid_argument("Error: faasConfig is nullptr!\n");


            if (!functionName || functionName->empty()) 
                throw std::invalid_argument("Error: functionName is invalid!\n") ;

            if (!faasConfig->hasFunction(functionName)) 
                throw std::invalid_argument("Warning: function \'" + *functionName + "\' not found in faasConfig.\n");
            
            std::call_once(init_flag, [&]() {
                ff_faas_connector::faasConfig = faasConfig;
            });
        }

        virtual std::unique_ptr<dataBuffer> invokeFaasFunction(const std::unique_ptr<dataBuffer> payload) = 0;

        virtual RegistrationResult registerFaasFunction() = 0;

        virtual DeRegistrationResult deregisterFaasFunction() = 0;

        std::shared_ptr<stats_entry> getStats() {            
            return stats;
        }

    protected: 
        std::shared_ptr<stats_entry> stats = nullptr; //stats for a single invocation
        inline static std::shared_ptr<const ff::ff_faas_config> faasConfig;
        const std::shared_ptr<std::string> functionName;
        inline static std::once_flag init_flag;
};

class ff_faas_connector_factory
{
    public:
    using Creator = std::function<std::unique_ptr<ff_faas_connector>(std::shared_ptr<const ff::ff_faas_config>, const std::shared_ptr<std::string>)>;

        static ff_faas_connector_factory& instance()
        {
            static ff_faas_connector_factory factory;
            return factory;
        }

        void registerConnector(const std::string& name, Creator creator)
        {
            std::lock_guard<std::mutex> lock(mtx);
            registry[name] = std::move(creator);

        }

        void deregisterConnector(const std::string& name)
        {

            std::lock_guard<std::mutex> lock(mtx);
            registry.erase(name);
        }

        std::unique_ptr<ff_faas_connector> create(const std::string& name, std::shared_ptr<const ff::ff_faas_config> conf, const std::shared_ptr<std::string> functionName) const
        {
            std::lock_guard<std::mutex> lock(mtx);
            auto it = registry.find(name);
            if (it != registry.end()) 
                return (it->second)(conf, functionName); 
                
            throw std::runtime_error("Unknown ff_faas_connector: " + name);
        }


    private:

        ff_faas_connector_factory() = default;
        ~ff_faas_connector_factory() = default;

        ff_faas_connector_factory(const ff_faas_connector_factory&) = delete;
        ff_faas_connector_factory& operator=(const ff_faas_connector_factory&) = delete;

        mutable std::mutex mtx;
        std::unordered_map<std::string, Creator> registry;
};

#endif /* FF_FAAS_CONNECTOR_HPP */