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
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <stdexcept>
#include <mutex>
#include <iostream>

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

    protected: 
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
            try {
                std::lock_guard<std::mutex> lock(mtx);
                registry[name] = std::move(creator);
            }
            catch (const std::system_error& e) {
                std::cerr << "Mutex error in registerConnector: " << e.what() << std::endl;
                throw;
            }
        }

        void deregisterConnector(const std::string& name)
        {
            try {
                std::lock_guard<std::mutex> lock(mtx);
                registry.erase(name);
            }
            catch (const std::system_error& e) {
                std::cerr << "Mutex error in deregisterConnector: " << e.what() << std::endl;
                throw;
            }
        }

        std::unique_ptr<ff_faas_connector> create(const std::string& name, std::shared_ptr<const ff::ff_faas_config> conf, const std::shared_ptr<std::string> functionName) const
        {
            try {
                std::lock_guard<std::mutex> lock(mtx);
                auto it = registry.find(name);
                if (it != registry.end()) {
                    return (it->second)(conf, functionName); 
                }
            }
            catch (const std::system_error& e) {
                std::cerr << "Mutex error in create: " << e.what() << std::endl;
                throw;
            }
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