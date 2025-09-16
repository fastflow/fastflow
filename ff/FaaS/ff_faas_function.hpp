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

namespace ff {

    template <typename IN_t, typename OUT_t = IN_t, typename T = void>
    class ff_faas_function {
    public:
        ff_faas_function(int argc = 0, char** argv = nullptr) {
            static_assert(std::is_base_of<ff_faas_function_adapter, T>::value, "T must be derived from ff_faas_function_adapter");
            PRINT_DBG(std::string("Inizializzazione della funzione FAAS con argc: ") + std::to_string(argc));
            
            if constexpr (traits::has_faas_alloctask_v<IN_t>) {
                this->alloctaskF = [](char* ptr, size_t sz) -> void* {
                    IN_t* p = nullptr;
                    traits::faas_alloctaskWrapper<IN_t>(ptr, sz, p);
                    return p;
                };
            }
            else {
                this->alloctaskF = [](char*, size_t) -> void* {
                    IN_t* o = new IN_t;
                    assert(o);
                    return o;
                };
            }

            if constexpr (traits::has_faas_freetask_v<OUT_t>) {
                this->freetaskF = [](void* o) {
                    traits::faas_freetaskWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o));
                };
            }
            else {
                this->freetaskF = [](void* o) {
                    if constexpr (!std::is_void_v<OUT_t>) {
                        OUT_t* obj = reinterpret_cast<OUT_t*>(o);
                        delete obj;
                    }
                };
            }

            if constexpr (traits::is_faas_serializable_v<OUT_t>) {
                this->serializeF = [](void* o, dataBuffer& b) -> bool {
                    bool datacopied = true;
                    std::pair<char*, size_t> p = traits::faas_serializeWrapper<OUT_t>(reinterpret_cast<OUT_t*>(o), datacopied);
                    b.setBuffer(p.first, p.second);
                    return datacopied;
                };
            }
            else {
                this->serializeF = [](void* data, dataBuffer& b) -> bool {
                    OUT_t* obj = static_cast<OUT_t*>(data);
                    bitsery::MeasureSize measureSize;
                    size_t neededSize = bitsery::quickSerialization<bitsery::MeasureSize>(measureSize, *obj);
                    auto buffer = new char[neededSize];
                    std::stringbuf sb(std::ios::in | std::ios::out | std::ios::binary);  
                    sb.pubsetbuf(buffer, neededSize);
                    std::ostream os(&sb);           
                    bitsery::Serializer<bitsery::OutputStreamAdapter> ser(os);
                    ser.object(*obj);  
                    b.setBuffer(buffer, neededSize, true);             
                    return true;
                };
            }
            PRINT_DBG("Funzione di serializzazione registrata.");

            if constexpr (traits::is_faas_deserializable_v<IN_t>) {
                this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                    IN_t* ptr = nullptr;
                    try{
                        ptr = static_cast<IN_t*>(this->alloctaskF(b.getPtr(), b.getLen()));
                    }
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during deserialization: " << e.what() << std::endl;
                        return nullptr;
                    }
                    datacopied = traits::faas_deserializeWrapper<IN_t>(b.getPtr(), b.getLen(), ptr);             
                    return ptr;
                };
            }
            else {
                this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                    IN_t* obj = nullptr;
                    try {
                         obj = reinterpret_cast<IN_t*>(this->alloctaskF(nullptr, 0));
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
            }
            PRINT_DBG("Funzione di deserializzazione registrata.");
            
            faas_fun_adapter = std::make_unique<T>(argc, argv);   
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
                unique_ptr<dataBuffer> InputParamsBuffer = faas_fun_adapter->ff_faas_function_adapter_getRequest(error_msg, stats_collection);
                
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
                IN_t* input_obj = static_cast<IN_t*>(deserializeF(*InputParamsBuffer, datacopied));
                if (!datacopied) 
                    InputParamsBuffer->doNotCleanup();   

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

                std::unique_ptr<dataBuffer> OutputParamsBuffer = nullptr;
                try {
                    OutputParamsBuffer = std::make_unique<dataBuffer>();
                    if (serializeF(output_obj, *OutputParamsBuffer))
                        freetaskF(output_obj);
                    else 
                        OutputParamsBuffer->freetaskF = freetaskF;
                } catch (const std::bad_alloc& e) {
                    if(!faas_fun_adapter->ff_faas_function_adapter_sendErrorResponse(std::make_unique<std::string>("Memory allocation failed during output serialization: " + std::string(e.what())))) {
                        std::cerr << "Failed to send error response to client." << std::endl;
                        return -1;
                    }
                    PRINT_DBG("Errore nella serializzazione dell'output.");
                    freetaskF(output_obj);
                    continue;
                }

                if(!faas_fun_adapter->ff_faas_function_adapter_sendResponse(std::move(OutputParamsBuffer), error_msg, T_fun_exec)) {
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
        bool stats_collection = false;
        std::function<void(void*)> freetaskF;
        std::function<void* (char*, size_t)> alloctaskF;
        std::function<bool(void*, dataBuffer&)> serializeF;
        std::function<void*(dataBuffer&, bool&)> deserializeF;
        std::unique_ptr<T> faas_fun_adapter;
    };

};

#endif /* FF_FAAS_FUNCTION_HPP */