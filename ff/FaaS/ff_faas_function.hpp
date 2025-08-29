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

#include <ff/ff_faas.hpp>
#include <bitsery/bitsery.h>
#include <bitsery/brief_syntax.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/adapter/stream.h>
#include <memory>

class ff_faas_function_adapter;
std::mutex debug_output_mutex;

namespace ff {
    class ff_faas_function_base {
        public:
            ff_faas_function_base() {}
            virtual ~ff_faas_function_base() = default;
            virtual std::unique_ptr<dataBuffer> handle_request(std::unique_ptr<dataBuffer> InputParamsBuffer) = 0;
    };

    template <typename IN_t, typename OUT_t = IN_t, typename T = void>
    class ff_faas_function: public ff::ff_faas_function_base{
    public:
        ff_faas_function(int argc = 0, char** argv = nullptr) {
            static_assert(std::is_base_of<ff_faas_function_adapter, T>::value, "T must be derived from ff_faas_function_adapter");
            PRINT_DBG("Inizializzazione della funzione FAAS con argc: " << argc);
            
            if constexpr (traits::has_alloctask_v<OUT_t>) {
                this->alloctaskF = [](char* ptr, size_t sz) -> void* {
                    OUT_t* p = nullptr;
                    alloctaskWrapper<OUT_t>(ptr, sz, p);
                    assert(p);
                    return p;
                };
            }
            else {
                this->alloctaskF = [](char*, size_t) -> void* {
                    OUT_t* o = new OUT_t;
                    assert(o);
                    return o;
                };
            }

            if constexpr (traits::has_freetask_v<IN_t>) {
                this->freetaskF = [](void* o) {
                    freetaskWrapper<IN_t>(reinterpret_cast<IN_t*>(o));
                };
            }
            else {
                this->freetaskF = [](void* o) {
                    if constexpr (!std::is_void_v<IN_t>) {
                        IN_t* obj = reinterpret_cast<IN_t*>(o);
                        delete obj;
                    }
                };
            }

            if constexpr (traits::is_serializable_v<IN_t>) {
                this->serializeF = [](void* o, dataBuffer& b) -> bool {
                    bool datacopied = true;
                    std::pair<char*, size_t> p = serializeWrapper<IN_t>(reinterpret_cast<IN_t*>(o), datacopied);
                    b.setBuffer(p.first, p.second);
                    return datacopied;
                };
            }
            else 
                if constexpr (traits::is_bitsery_serializable<IN_t>::value) {
                    this->serializeF = [](void* data, dataBuffer& b) -> bool {
                        IN_t* obj = static_cast<IN_t*>(data);
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
            else {
                static_assert(traits::always_false_v<IN_t>,"Type IN_t is not serializable by any known method.");
            }
            PRINT_DBG("Funzione di serializzazione registrata.");

            if constexpr (traits::is_deserializable_v<IN_t>) {
                this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                    IN_t* ptr = static_cast<OUT_t*>(this->alloctaskF(b.getPtr(), b.getLen()));
                    datacopied = deserializeWrapper<OUT_t>(b.getPtr(), b.getLen(), ptr);
                    return ptr;
                };
            }
            else 
                if constexpr (traits::is_bitsery_deserializable<IN_t>::value) {
                    this->deserializeF = [this](dataBuffer& b, bool& datacopied) -> void* {
                        IN_t* obj = reinterpret_cast<IN_t*>(this->alloctaskF(nullptr, 0));
                        std::istream is(&b);
                        bitsery::Deserializer<bitsery::InputStreamAdapter> des(is);
                        des.object(*obj);

                        if (des.adapter().error() != bitsery::ReaderError::NoError) {
                            this->freetaskF(obj);
                            return nullptr;
                        }

                        datacopied = true;
                        return obj;
                };
            }
            else 
                static_assert(traits::is_deserializable_v<OUT_t> || traits::is_bitsery_deserializable<OUT_t>::value,
                            "Type OUT_t is not deserializable by any known method.");

            PRINT_DBG("Funzione di deserializzazione registrata.");
            
            faas_fun_adapter = std::make_unique<T>(*this, argc, argv);   
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
            PRINT_DBG("Esecuzione della funzione FAAS...");
            if(!faas_fun_adapter->ff_faas_function_adapter_run()) {
                PRINT_DBG("Errore nell'esecuzione dell'adapter.");
                return -1;
            }
            PRINT_DBG("Funzione FAAS completata.");
            return 0;    
        }

        std::unique_ptr<dataBuffer> handle_request(std::unique_ptr<dataBuffer> InputParamsBuffer) {
            PRINT_DBG("Gestione richiesta...");

            bool datacopied = false;
            IN_t* input_obj = static_cast<IN_t*>(deserializeF(*InputParamsBuffer, datacopied));
            if (!datacopied) 
                InputParamsBuffer->doNotCleanup();  // Non fare cleanup del buffer, poiché è già gestito dal deserializzatore  

            if(!input_obj) {
                PRINT_DBG("Errore nella deserializzazione dell'input.");
                return nullptr;
            }

            PRINT_DBG("Input deserializzato con successo.");
            OUT_t* output_obj = svc(input_obj);
            
            std::unique_ptr<dataBuffer> OutputParamsBuffer = std::make_unique<dataBuffer>();
            if (serializeF(output_obj, *OutputParamsBuffer)) {
                PRINT_DBG("Serializzazione dell'output riuscita.");
                freetaskF(output_obj);
            }
            else {
                OutputParamsBuffer->freetaskF = freetaskF;
                PRINT_DBG("Serializzazione dell'output fallita.");
            }
            return OutputParamsBuffer;
        }

    private:
        std::function<void(void*)> freetaskF;
        std::function<void* (char*, size_t)> alloctaskF;
        std::function<bool(void*, dataBuffer&)> serializeF;
        std::function<void*(dataBuffer&, bool&)> deserializeF;
        std::unique_ptr<T> faas_fun_adapter;
    };

};

#endif /* FF_FAAS_FUNCTION_HPP */