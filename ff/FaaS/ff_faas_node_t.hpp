/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file faas_node.hpp
 *  \ingroup building_blocks
 *
 *  \brief  Defines the FastFlow support for the offloading to a FAAS framework
 *
 *  It contains the definition of the \p ff_faas_node_t class, which is an
 *  extension of the base class \p ff_node_t, with features oriented
 *  offloading the svc function to FAAS systems.
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

#ifndef FF_FAAS_NODE_HPP
#define FF_FAAS_NODE_HPP

#include <ff/FaaS/ff_faas_function_invoker_thread.hpp>
#include <bitsery/bitsery.h>
#include <bitsery/brief_syntax.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/adapter/stream.h>
#include <string>
#include <memory>
#include <mutex>
#include <vector>

namespace ff {

    template <typename IN_t, typename OUT_t = IN_t>
    class ff_faas_node_t : public ff_node_t<IN_t, OUT_t> {
    public:
        using typename ff_node_t<IN_t, OUT_t>::in_type;
        using typename ff_node_t<IN_t, OUT_t>::out_type;

        ff_faas_node_t(std::shared_ptr<std::string> fName,                       
                       const std::string& backendFile = std::string(DEFAULT_BACKEND_FILE),
                       const std::string& functionsFile = std::string(DEFAULT_FUNCTIONS_FILE), 
                       size_t initial_parallelism_degree = 0)
            : num_task(0),stats_collection(false), functionName(fName), sleeping(false) 
        {
            PRINT_DBG("Constructor called for master node thread.");
            if (!functionName || functionName->empty()) 
                throw std::runtime_error("Function name cannot be empty.\n");

            std::call_once(init_flag, [&]() {
                PRINT_DBG(std::string("Reading configuration for ") + *functionName);
                faasConfig = std::make_shared<ff_faas_config>(backendFile, functionsFile);
                PRINT_DBG(std::string("Configuration for ") + *functionName + std::string(" read."));
            });

            if (initial_parallelism_degree==0)
                parallelism_degree = faasConfig->getFunctionInitialParallelism(functionName);
            else
                parallelism_degree = initial_parallelism_degree;

            stats_collection = faasConfig->isStatsCollectionEnabled(functionName);
            PRINT_DBG(std::string("Initial parallelism for ") + *functionName + std::string(" is ") + std::to_string(parallelism_degree));        

            if (!faasConfig || !faasConfig->hasFunction(functionName)) 
                throw std::runtime_error("Function \'" + *functionName + "\' not found in configuration file.\n");
            
            if constexpr (traits::has_faas_allocTask_member<OUT_t>::value) {
                this->faas_alloctaskF = reinterpret_cast<void*(*)(char*,size_t)>(OUT_t::faas_alloc);
            }
            else {
                this->faas_alloctaskF = [](char*, size_t) -> void* {
                    OUT_t* o = new OUT_t;
                    assert(o);
                    return o;
                };
            }
            PRINT_DBG(std::string("Allocation function for function ") + *functionName + std::string(" read."));

            if constexpr (traits::has_faas_freeTask_member<IN_t>::value) {
                this->faas_freetaskF = [](void* o) {
                    IN_t::faas_freeTask(reinterpret_cast<IN_t*>(o));
                };

            }
            else
                this->faas_freetaskF = [](void* o) { delete reinterpret_cast<IN_t*>(o); };                            

            PRINT_DBG(std::string("Deallocation function for function ") + *functionName + std::string(" read."));

            if constexpr (traits::has_faas_serialize_member<IN_t>::value) {
                this->faas_serializeF = [](void* o, faasBuffer& b) -> bool {
                    auto [buff, size, datacopied] = reinterpret_cast<IN_t*>(o)->faas_serialize();
                    b.setBuffer(buff,size,datacopied);
                    return datacopied;
                };
                PRINT_DBG("Serialization function manually serializable!");
            }
            else {
                this->faas_serializeF = [](void* data, faasBuffer& b) -> bool {
                    IN_t* obj = static_cast<IN_t*>(data);
                    bitsery::MeasureSize measureSize;
                    size_t neededSize = bitsery::quickSerialization<bitsery::MeasureSize,IN_t>(measureSize, *obj);
                    b.reuseBuffer(neededSize);
                    std::ostream os(&b);           
                    bitsery::Serializer<bitsery::OutputStreamAdapter> ser(os);
                    ser.object(*obj);  
                    return true;
                };
                PRINT_DBG("Serialization function is Bitsery serializable!"); 
            }
            PRINT_DBG(std::string("Serialization function for function ") + *functionName + std::string(" read."));

            if constexpr (traits::has_faas_deserialize_member<OUT_t>::value) {
                //TODO: da rivedere! Non capisco perché nel codice di Tonci c'è un terzo parametro
                this->faas_deserializeF = [this](faasBuffer& b, bool& datacopied) -> void* {
                    OUT_t* ptr = nullptr;
                    try{
                        ptr = static_cast<OUT_t*>(this->faas_alloctaskF(b.getBuffer(), b.size()));
                    }
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during deserialization: " << e.what() << " for function: " << *functionName << std::endl;
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
                    OUT_t* obj = nullptr;
                    try {
                         obj = reinterpret_cast<OUT_t*>(this->faas_alloctaskF(nullptr, 0));
                    } 
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during deserialization: " << e.what() << " for function: " << *functionName << std::endl;
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
                
            PRINT_DBG(std::string("Deserialization function for function ") + *functionName + std::string(" read."));
            
            for (unsigned long i = 0; i < parallelism_degree; ++i) {     
                // Create a new invoker thread
                std::unique_ptr<ff_faas_function_invoker_thread<IN_t, OUT_t>> thread = std::make_unique<ff_faas_function_invoker_thread<IN_t, OUT_t>>(functionName, faasConfig, *this);                 
                thread->start();  // Start the thread
                faas_function_invoker_threads.push_back(std::move(thread));  // Store the thread in the vector
            }      

            if(stats_collection) 
                stats_map = std::make_shared<std::unordered_map<unsigned long, std::shared_ptr<stats_entry>>>();            
        }

        virtual ~ff_faas_node_t() = default;

        size_t getParallelismDegree() {            
            return parallelism_degree.load(std::memory_order_relaxed);
        }

        size_t getCurrentParallelismDegree() { 
            std::unique_lock<std::mutex> lock(change_degree_mtx); // Ensure thread-safety    
            return faas_function_invoker_threads.size();
        }

        bool setParallelismDegree(size_t new_degree) { 
            PRINT_DBG(std::string("Setting new parallelism degree for function ") + *functionName + std::string(" read."));    
            if (new_degree <= 0) {
                std::cerr << "Parallelism degree must be positive." << std::endl;
                return false;
            }
            parallelism_degree.store(new_degree,std::memory_order_release);
            PRINT_DBG(std::string("New parallelism degree for function ") + *functionName + std::string(" set."));    
            return true;
        }

        // It returns the statistics until the actual moment
        std::shared_ptr<std::unordered_map<unsigned long, std::shared_ptr<stats_entry>>> getRealTimeStats() {
            if(stats_collection) {                
                std::lock_guard<std::mutex> lock(stats_map_mtx);
                return stats_map;
            }
            return nullptr;
        }

    protected:

        bool ff_send_out(void* task, int id = -1, unsigned long retry = ((unsigned long)-1), unsigned long ticks = (ff_node::TICKS2WAIT)) override final {
            try {
                std::unique_lock<std::mutex> lock(send_out_mtx);
                return ff_node_t<IN_t, OUT_t>::ff_send_out(task, id, retry, ticks);
            }   catch (const std::exception& e) {
                std::cerr << "Unexpected exception: " << e.what() << std::endl;                
            } catch (...) {
                std::cerr << "Unknown exception in ff_send_out" << std::endl;                
            }
            return false;
        }

        inline void worker_thaw() {
            PRINT_DBG(std::string("Trying to resume main thread for function ") + *functionName);    
            if (sleeping.load(std::memory_order_acquire)) {                                
                std::lock_guard<std::mutex> lock(sleep_mtx);
                PRINT_DBG("Main thread is sleeping, notifying it to wake up.");
                cv.notify_one();                
            }            
        }

        bool insert_stats_entry(unsigned long task_id, std::shared_ptr<stats_entry> entry) {            
            try {
                std::lock_guard<std::mutex> lock(stats_map_mtx);
                (*stats_map)[task_id] = entry;            
            } catch (const std::bad_alloc& e) {
                std::cerr << "Memory allocation failed during stats insertion: " << e.what() << " for function: " << *functionName << std::endl;
                return false;
            }
            catch (const std::exception& e) {
                std::cerr << "Unexpected exception in stats insertion: " << e.what() << std::endl;
                return false;
            } catch (...) {
                std::cerr << "Unknown error in stats insertion" << std::endl;
                return false;   
            }
            return true;
        }

        using ff_node_t<IN_t,OUT_t>::GO_ON;
        using ff_node_t<IN_t,OUT_t>::ff_send_out;

        struct internal_task_t{
            IN_t* task;
            unsigned long task_id;
            std::unique_ptr<std::chrono::high_resolution_clock::time_point> T_total_start;
        };

        friend class ff_faas_function_invoker_thread<IN_t, OUT_t>;
    private:

        ff_faas_node_t(const ff_faas_node_t& other) = delete;
        ff_faas_node_t& operator=(const ff_faas_node_t& other) = delete;

        void eosnotify(ssize_t) override final {
            PRINT_DBG(std::string("EOS arrived to main thread for ") + *functionName);    
            std::unique_lock<std::mutex> lock(change_degree_mtx);
            for (auto& thread : faas_function_invoker_threads) {
                PRINT_DBG(std::string("Stopping worker thread: "),thread->get_id());  
                try{
                    thread->stop();                                
                } catch(...) {
                    std::cerr << "Exception while stopping the worker thread for function: " << *functionName << std::endl;
                    thread->detach_worker();
                }   
                PRINT_DBG("Worker thread stopped");
            }
            faas_function_invoker_threads.clear();       
        }

        OUT_t* svc(IN_t* task) override final {
            internal_task_t* internal_task = new internal_task_t();
            internal_task->task = task;            
            if(stats_collection) {
                // id of the task, for statistics collection purposes
                internal_task->task_id = num_task;
                try{
                    internal_task->T_total_start = std::make_unique<std::chrono::high_resolution_clock::time_point>(std::chrono::high_resolution_clock::now());
                } catch (const std::bad_alloc& e) {
                    std::cerr << "Memory allocation failed during task creation: " << e.what() << " for function: " << *functionName << std::endl;
                    return GO_ON;
                }                    
                ++num_task;
            }

            while (true) {
                size_t active_thread_count = faas_function_invoker_threads.size();
                unsigned long current_parallelism_degree = parallelism_degree.load(std::memory_order_acquire);
                if (active_thread_count < current_parallelism_degree) {
                    PRINT_DBG(std::string("Parallelism too low. We need a number of threads equal to ") + std::to_string((current_parallelism_degree - active_thread_count)));    
                    for (size_t i = 0; i < current_parallelism_degree - active_thread_count; ++i) {
                        bool task_assigned = false;
                        try {
                            auto thread = std::make_unique<ff_faas_function_invoker_thread<IN_t, OUT_t>>(functionName, faasConfig, *this);                        
                            thread->start();
                            if(!task_assigned) {                                                            
                                thread->push_input(internal_task);
                                task_assigned = true;
                            }
                            faas_function_invoker_threads.push_back(std::move(thread));  
                        } catch (const std::bad_alloc& e) {
                            std::cerr << "Memory allocation failed during thread creation: " << e.what() << " for function: " << *functionName << std::endl;
                        } catch (...) {
                            std::cerr << "Exception while creating the worker thread for function: " << *functionName << std::endl;
                        }
                    }                    
                    return GO_ON;
                }
                else 
                    if (active_thread_count > current_parallelism_degree) {
                        size_t threads_to_stop = active_thread_count - current_parallelism_degree;
                        PRINT_DBG(std::string("Parallelism too high. We need to stop a number of threads equal to ") + std::to_string(threads_to_stop));    
                        for (auto it = faas_function_invoker_threads.begin(); it != faas_function_invoker_threads.end();) { 
                            auto thread = it->get();
                            if(threads_to_stop > 0) {
                                try{
                                    if(thread->try_stop()) {
                                        std::unique_lock<std::mutex> lock(change_degree_mtx);
                                        it = faas_function_invoker_threads.erase(it);
                                        threads_to_stop--;      
                                        continue;              
                                    }
                                }
                                catch(...) {
                                    std::cerr << "Exception while stopping the worker thread for function: " << *functionName << std::endl;
                                    thread->detach_worker();
                                    it = faas_function_invoker_threads.erase(it);
                                    threads_to_stop--;                    
                                    continue;
                                }
                            }
                            
                            if (thread->push_input(internal_task)) 
                                return GO_ON;   
                            ++it;                                          
                        }
                    }

                for (auto& thread : faas_function_invoker_threads) 
                    if (thread->push_input(internal_task)) 
                        return GO_ON;
                
                {
                    std::unique_lock<std::mutex> lock(sleep_mtx);

                    sleeping.store(true,std::memory_order_release);
                    PRINT_DBG("Main thread goes to sleep, waiting for free worker threads");                        

                    cv.wait(lock, [&]() {
                        for (auto& thread : faas_function_invoker_threads) {
                            if (thread->can_accept_task())
                                return true;
                        }
                        return false;});
                    PRINT_DBG("Main thread woke up, some worker is free now");
                    sleeping.store(false,std::memory_order_release);
                }
            }
        }

        unsigned long num_task;
        bool stats_collection;
        alignas(CACHE_LINE_SIZE) std::mutex stats_map_mtx;
        std::shared_ptr<std::unordered_map<unsigned long, std::shared_ptr<stats_entry>>> stats_map;
        std::shared_ptr<std::string> functionName;  
        alignas(CACHE_LINE_SIZE) std::atomic<size_t> parallelism_degree;
        alignas(CACHE_LINE_SIZE) std::atomic<bool> sleeping;
        alignas(CACHE_LINE_SIZE) inline static  std::shared_ptr<const ff_faas_config> faasConfig = nullptr;
        alignas(CACHE_LINE_SIZE) inline static  std::once_flag init_flag;
        alignas(CACHE_LINE_SIZE) std::mutex change_degree_mtx;
        alignas(CACHE_LINE_SIZE) std::mutex sleep_mtx;
        alignas(CACHE_LINE_SIZE) std::mutex send_out_mtx;
        alignas(CACHE_LINE_SIZE) std::condition_variable cv;
        std::vector<std::unique_ptr<ff_faas_function_invoker_thread<IN_t, OUT_t>>> faas_function_invoker_threads;

        std::function<void(void*)> faas_freetaskF;
        std::function<void* (char*, size_t)> faas_alloctaskF;
        std::function<bool(void*, faasBuffer&)> faas_serializeF;
        std::function<void*(faasBuffer&, bool&)> faas_deserializeF;
    };
} // namespace ff

#endif /* FF_FAAS_NODE_HPP */