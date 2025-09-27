/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file ff_faas_function.hpp
 *  \ingroup ff
 *
 *  \brief  Defines the ff_faas_function class to be used as an helper class
 *          to define C++ functions that can be offloaded to a FAAS framework. 
 *
 *  It contains the definition of the \p ff_faas_function class, an helper class to define
 *  C++ function that can be offloaded to a FAAS framework.
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

#ifndef FF_FAAS_FUNCTION_HPP
#define FF_FAAS_FUNCTION_HPP

#include <ff/ff_faas_configuration.hpp>
#include <ff/FaaS/ff_faas_typetraits.hpp>
#include <ff/FaaS/ff_faas_function_adapter.hpp>
#include <bitsery/bitsery.h>
#include <bitsery/brief_syntax.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/adapter/stream.h>
#include <memory>
#include <chrono>

using namespace adapter;

namespace ff {

    template <typename IN_t, typename OUT_t = IN_t, typename T = void>
    class ff_faas_function {
    public:
        ff_faas_function(int argc = 0, char** argv = nullptr): stats_collection(false) {
            static_assert(std::is_base_of<ff_faas_function_adapter, T>::value, "T must be derived from ff_faas_function_adapter");
            PRINT_DBG(std::string("Inizializzazione della funzione FAAS con argc: ") + std::to_string(argc));

            if constexpr (traits::has_faas_allocTask_member<IN_t>::value) {
                this->faas_alloctaskF = reinterpret_cast<void*(*)(char*,size_t)>(IN_t::faas_alloc);
            }
            else {
                this->faas_alloctaskF = [](char*, size_t) -> void* {
                    IN_t* o = new IN_t;
                    assert(o);
                    return o;
                };
            }
            PRINT_DBG("Allocation function read.");

            if constexpr (traits::has_faas_freeTask_member<OUT_t>::value) {
                this->faas_freetaskF = [](void* o) {
                    OUT_t::freeTask(reinterpret_cast<OUT_t*>(o));
                };
            }
            else
                this->faas_freetaskF = [](void* o) { delete reinterpret_cast<OUT_t*>(o); };                            

            PRINT_DBG("Deallocation function read.");

            if constexpr (traits::has_faas_serialize_member<OUT_t>::value) {
                this->faas_serializeF = [](void* o, faasBuffer& b) -> bool {
                    auto [buff, size, datacopied] = reinterpret_cast<OUT_t*>(o)->faas_serialize();
                    b.setBuffer(buff,size,datacopied);
                    return datacopied;
                };
                PRINT_DBG("Serialization function manually serializable!");
            }
            else {
                this->faas_serializeF = [](void* data, faasBuffer& b) -> bool {
                    OUT_t* obj = static_cast<OUT_t*>(data);
                    bitsery::MeasureSize measureSize;
                    size_t neededSize = bitsery::quickSerialization<bitsery::MeasureSize,OUT_t>(measureSize, *obj);
                    b.reuseBuffer(neededSize);
                    std::ostream os(&b);           
                    bitsery::Serializer<bitsery::OutputStreamAdapter> ser(os);
                    ser.object(*obj);  
                    return true;
                };
                PRINT_DBG("Serialization function is Bitsery serializable!"); 
            }
            PRINT_DBG(std::string("Serialization function read."));

            if constexpr (traits::has_faas_deserialize_member<IN_t>::value) {
                //TODO: da rivedere! Non capisco perché nel codice di Tonci c'è un terzo parametro
                this->faas_deserializeF = [this](faasBuffer& b, bool& datacopied) -> void* {
                    IN_t* ptr = nullptr;
                    try{
                        ptr = static_cast<IN_t*>(this->faas_alloctaskF(b.getBuffer(), b.size()));
                    }
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during deserialization: " << e.what() << std::endl;
                        return nullptr;
                    }
                    datacopied = ptr->faas_deserialize(b.getBuffer(), b.size());            
                    return ptr;
                };
                PRINT_DBG("Serialization function manually deserializable!");
            }
            else {
                //TODO: da rivedere! Non capisco perché nel codice di Tonci c'è un terzo parametro
                this->faas_deserializeF = [this](faasBuffer& b, bool& datacopied) -> void* {
                    IN_t* obj = nullptr;
                    try {
                         obj = reinterpret_cast<IN_t*>(this->faas_alloctaskF(nullptr, 0));
                    } 
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during deserialization: " << e.what() << std::endl;
                        return nullptr;
                    }
                    // we supposed no error in deserialization
                    std::istream is(&b);
                    bitsery::Deserializer<bitsery::InputStreamAdapter> des(is);
                    des.object(*obj);
                    datacopied = true;
                    return obj;
                };
                PRINT_DBG("Serialization function is Bitsery deserializable!");
            }

            faas_fun_adapter = std::make_unique<T>(argc, argv);
            OutputParamsBuffer = std::make_shared<faasBuffer>();   
            PRINT_DBG("Adapter FAAS creato.");
            
            if(!faas_fun_adapter->ff_faas_function_adapter_init()) {
                PRINT_DBG("Errore nell'inizializzazione dell'adapter FAAS.");
                throw std::runtime_error("Failed to initialize the FAAS function adapter.");
            }
            PRINT_DBG("Inizializzazione dell'adapter FAAS completata.");
        }

        virtual ~ff_faas_function() = default;

        virtual OUT_t* svc(IN_t* data) = 0;

        int run_and_wait_end() {
            while(true) {
                unique_ptr<std::string> error_msg = nullptr;
                PRINT_DBG("Attesa richiesta dal client nella funzione FAAS...");
                unique_ptr<ff::faasBuffer> InputParamsBuffer = faas_fun_adapter->ff_faas_function_adapter_getRequest(error_msg, stats_collection);
                
                if(!InputParamsBuffer) {
                    if(!faas_fun_adapter->ff_faas_function_adapter_sendErrorResponse(std::move(error_msg))) {
                        std::cerr << "Failed to send error response to client." << std::endl;
                        return -1;
                    }
                    PRINT_DBG("Errore nell'ottenere la richiesta.");
                    continue;
                }

                PRINT_DBG("Richiesta ottenuta correttamente. Gestione richiesta...");

                bool datacopied = false;
                IN_t* input_obj = static_cast<IN_t*>(faas_deserializeF(*InputParamsBuffer, datacopied));                
                InputParamsBuffer->setCleanup(datacopied);   

                if(!input_obj) {
                    if(!faas_fun_adapter->ff_faas_function_adapter_sendErrorResponse(std::make_unique<std::string>("Error during input deserialization."))) {
                        std::cerr << "Failed to send error response to client." << std::endl;
                        return -1;
                    }
                    PRINT_DBG("Errore nella deserializzazione dell'input.");
                    continue;
                }

                PRINT_DBG("Input deserializzato con successo.");
                
                std::chrono::high_resolution_clock::time_point T_fun_exec_start;
                if(stats_collection) 
                    T_fun_exec_start = std::chrono::high_resolution_clock::now();
                
                OUT_t* output_obj = svc(input_obj);
                    
                double T_fun_exec = -1.0;
                if(stats_collection) {
                    std::chrono::duration<double> T_fun_exec_dur = std::chrono::high_resolution_clock::now() - T_fun_exec_start;
                    T_fun_exec = std::chrono::duration<double, std::micro>(T_fun_exec_dur).count();
                    PRINT_DBG(std::string("Funzione eseguita in: ") + std::to_string(T_fun_exec) + std::string(" microsecondi"));
                }

                try {
                    if (faas_serializeF(output_obj, *OutputParamsBuffer))
                        faas_freetaskF(output_obj);
                    else 
                        OutputParamsBuffer->freetaskF = faas_freetaskF;
                } catch (const std::bad_alloc& e) {
                    if(!faas_fun_adapter->ff_faas_function_adapter_sendErrorResponse(std::make_unique<std::string>("Memory allocation failed during output serialization: " + std::string(e.what())))) {
                        std::cerr << "Failed to send error response to client." << std::endl;
                        return -1;
                    }
                    PRINT_DBG("Errore nella serializzazione dell'output.");
                    faas_freetaskF(output_obj);
                    continue;
                }

                if(!faas_fun_adapter->ff_faas_function_adapter_sendResponse(OutputParamsBuffer, error_msg, T_fun_exec)) {
                    if(!faas_fun_adapter->ff_faas_function_adapter_sendErrorResponse(std::move(error_msg))) {
                        std::cerr << "Failed to send error response to client." << std::endl;
                        return -1;
                    }
                    PRINT_DBG("Errore nell'invio della risposta al client.");
                    continue;
                }                    

                PRINT_DBG("Funzione FAAS completata.");
            }
        }

    private:
        std::shared_ptr<faasBuffer> OutputParamsBuffer;
        bool stats_collection;
        std::function<void(void*)> faas_freetaskF;
        std::function<void* (char*, size_t)> faas_alloctaskF;
        std::function<bool(void*, faasBuffer&)> faas_serializeF;
        std::function<void*(faasBuffer&, bool&)> faas_deserializeF;
        std::unique_ptr<T> faas_fun_adapter;
    };

};

#endif /* FF_FAAS_FUNCTION_HPP */