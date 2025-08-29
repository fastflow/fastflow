/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file ff_faas_function_invoker_thread.hpp
 *  \ingroup ff
 *
 *  \brief  Defines the ff_faas_function_invoker_thread class to offload and invoke a 
            function on a specific FAAS framework
 *
 *  It contains the definition of the \p ff_faas_function_invoker_thread class,
 *  with features oriented invoking a single function on FAAS systems.
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

#ifndef FF_FAAS_FUNCTION_INVOCER_THREAD_HPP
#define FF_FAAS_FUNCTION_INVOCER_THREAD_HPP

#include <ff/FaaS/ff_faas_connector.hpp>
#include <atomic>
#include <string>
#include <functional>
#include <memory>
#include <vector>
#include <cassert>
#include <mutex>
#include <condition_variable>
#include <thread>

namespace ff {

    template<typename IN_t, typename OUT_t>
    class ff_faas_node_t;  // Forward declaration

    template<typename IN_t, typename OUT_t>
    class ff_faas_function_invoker_thread {
    public:
        ff_faas_function_invoker_thread(const std::shared_ptr<std::string> functionName,std::shared_ptr<const ff::ff_faas_config> faasConfig, 
                                        ff_faas_node_t<IN_t, OUT_t>& faas_node)
            : functionName(functionName),faas_node(faas_node),task(nullptr),stop_flag(false),sleeping(false) {
                const std::string& faasType = faasConfig->getFunctionFaasType(functionName);

                faasConnector = ff_faas_connector_factory::instance().create(faasType, faasConfig, functionName);
                if (!faasConnector) 
                    throw std::runtime_error("Failed to create connector for type \'" + faasType + "\'\n");

                if (faasConnector->registerFaasFunction() == ff_faas_connector::REGISTRATION_ERROR) 
                    throw std::runtime_error("Failed to register FaaS function: " + *functionName + "\n");

                stats_collection = faasConfig->isStatsCollectionEnabled(functionName);

                PRINT_DBG("Constructor called for function invoker thread for function: " + *functionName);
        }

        ~ff_faas_function_invoker_thread(){
            try{ 
                PRINT_DBG("Destructor called for function invoker thread for function: " + *functionName);
                if (faasConnector) {
                    if (faasConnector->deregisterFaasFunction() == ff_faas_connector::DEREGISTRATION_ERROR) 
                        std::cerr << "Failed to deregister FaaS function: " << *functionName << std::endl;                    
                }
            } catch(...) {
                std::cerr << "Exception in deregister FaaS function: " << *functionName << std::endl;
            }    
        };

        #ifdef DEBUG
        inline std::thread::id get_id() const {
            return worker_thread.get_id();
        }
        #endif

        void inline start() {
            worker_thread = std::thread(&ff_faas_function_invoker_thread::run, this);
            PRINT_DBG("Worker thread started.");
        }

        bool try_stop() {
            PRINT_DBG("Trying to stop the worker thread...");

            // Stop a thread only if it is doing nothing     
            auto data = task.load(std::memory_order_acquire);

            if(data == nullptr) {
                stop();                
                return true;  // Successfully stopped
            }
            PRINT_DBG("Worker thread not stopped, task is still running.");
            return false;  // Thread is busy, cannot stop
        }

        void stop() {
            PRINT_DBG("Stopping the worker thread...");
            stop_flag.store(true,std::memory_order_release);
            // Notify the worker thread to wake up if it's sleeping
            if (sleeping.load(std::memory_order_acquire)) {
                PRINT_DBG("Worker thread is sleeping, notifying it to wake up.");
                cv.notify_one();  // Only notify if the consumer is actually sleeping
            }
            if (worker_thread.joinable()) {
                worker_thread.join();  // Wait for the thread to complete
            }
            PRINT_DBG("Worker thread stopped successfully.");
        }

        void detach_worker() {
            if (worker_thread.joinable()) {
                worker_thread.detach();
            }
        }

        bool inline can_accept_task() const {
            return task.load(std::memory_order_acquire) == nullptr;
        }

        bool push_input(ff_faas_node_t<IN_t,OUT_t>::internal_task_t* data) {
            PRINT_DBG("Trying to send a new task to the worker thread...");

            typename ff_faas_node_t<IN_t,OUT_t>::internal_task_t* expected = nullptr;

            // The producer returns immediately if data is NOT nullptr, 
            // the consumer is either running or it is about to reset the task
            if (!task.compare_exchange_strong(expected, data, std::memory_order_release, std::memory_order_relaxed)) {
                PRINT_DBG("Worker thread is busy, cannot accept new task.");
                return false;  // If task is not nullptr, it means consumer is still processing
            }

            // Notify the worker thread (consumer) that there is new data to process
            if (sleeping.load(std::memory_order_acquire)) {
                PRINT_DBG("Worker thread is sleeping, notifying it to wake up.");
                cv.notify_one();  // Only notify if the consumer is actually sleeping
            }
            PRINT_DBG("New task sent to the worker thread successfully.");
            return true;
        }

    private:

        ff_faas_function_invoker_thread(const ff_faas_function_invoker_thread& other) = delete;
        ff_faas_function_invoker_thread& operator=(const ff_faas_function_invoker_thread& other) = delete;

        void serialize(IN_t* input, std::unique_ptr<dataBuffer>& serializedData) {
            try{
                PRINT_DBG("Serializing input data for function: " + *functionName);
                if (faas_node.serializeF(input, *serializedData))
                    faas_node.freetaskF(input);
                else 
                    serializedData->freetaskF = faas_node.freetaskF;
                PRINT_DBG("Input data serialized for function: " + *functionName);
            }
            catch (...) {
                std::cerr << "Exception received during serializing task for function: " << *functionName << std::endl;
                PRINT_DBG("Input data not serialized for function: " + *functionName);
                faas_node.freetaskF(input);
            }
        }

        OUT_t* deserialize(std::unique_ptr<dataBuffer> resultBuffer) {
            PRINT_DBG("Deserializing output data for function: " + *functionName);
            bool datacopied = false;
            OUT_t* task = static_cast<OUT_t*>(faas_node.deserializeF(*resultBuffer, datacopied));            
            if (!datacopied) 
                resultBuffer->doNotCleanup();  // Do not cleanup the buffer, as it is already handled by the deserializer         
            PRINT_DBG("Output data deserialized for function: " + *functionName);
            return task;      
        }

        void run() {
            while (true) {
                unsigned long task_id = 0;
                std::unique_ptr<std::chrono::high_resolution_clock::time_point> T_total_start = nullptr;
                typename ff_faas_node_t<IN_t,OUT_t>::internal_task_t* internal_task = nullptr;
                PRINT_DBG("Worker thread is running: started a new cycle.");

                // Load the task from the atomic pointer with memory_order_acquire to ensure visibility
                internal_task = task.load(std::memory_order_acquire);

                if (internal_task != nullptr) {
                    PRINT_DBG("Worker thread is processing a task for function: " + *functionName);
                    // Process the data if it's not nullptr
                    std::unique_ptr<dataBuffer> serializedData = nullptr;
                    try {
                        IN_t* data = internal_task->task;
                        if(stats_collection) {
                            T_total_start = std::move(internal_task->T_total_start);
                            task_id = internal_task->task_id;
                        }             
                        delete internal_task;  // Free the internal task structure           
                        serializedData = std::make_unique<dataBuffer>();                                           
                        serialize(data, serializedData);
                    }
                    catch (const std::bad_alloc& e) {
                        std::cerr << "Memory allocation failed during serialization: " << e.what() << " for function: " << *functionName << std::endl;
                    }

                    if (serializedData) {
                        std::unique_ptr<dataBuffer> resultBuffer = faasConnector->invokeFaasFunction(std::move(serializedData));
                        if (resultBuffer) {
                            OUT_t* new_task = deserialize(std::move(resultBuffer));

                            if ((new_task != nullptr)&&(faas_node.get_out_buffer()!=nullptr)) {   
                                // we suppose it cannot failed forever                                         
                                while(!faas_node.ff_send_out(new_task)) {}                                
                                PRINT_DBG("Worker thread sent output task for function: " + *functionName);
                                if(stats_collection) {
                                    try{
                                        std::shared_ptr<stats_entry> stats = faasConnector->getStats();
                                        if(stats) {                                                                                          
                                            std::chrono::duration<double> T_total_dur = std::chrono::high_resolution_clock::now() - *T_total_start;
                                            stats->T_total = std::chrono::duration<double, std::micro>(T_total_dur).count();   
                                            stats->T_ff_overhead = stats->T_total - stats->T_faas_overhead - stats->T_fun_exec - stats->T_comm;                            
                                            faas_node.insert_stats_entry(task_id, stats);
                                        }
                                    } catch (const std::bad_alloc& e) {
                                        std::cerr << "Memory allocation failed during statistics collection: " << e.what() << " for function: " << *functionName << std::endl;                                        
                                    }                                    
                                }
                            }
                        }
                        else 
                            std::cerr << "Error invoking FaaS function: " << *functionName << std::endl;                    
                    } else 
                        std::cerr << "Error serializing task for function: " << *functionName << std::endl;

                    // After processing, reset the task to nullptr, also in case of an error
                    task.store(nullptr, std::memory_order_release);    
                    PRINT_DBG("Worker thread reset task to nullptr for function: " + *functionName);
                    faas_node.worker_thaw();
                    PRINT_DBG("Worker thread finished processing task for function: " + *functionName);                
                    if (stop_flag.load(std::memory_order_acquire)) {
                        PRINT_DBG("Worker thread stopping as stop flag is set.");
                        return;  // Stop the loop if the flag is set
                    }

                } else {
                    // If task is nullptr, the consumer needs to sleep 
                    // Check the stop flag
                    if (stop_flag.load(std::memory_order_acquire)) {
                    PRINT_DBG("Worker thread stopping as stop flag is set.");
                        return;  // Stop the loop if the flag is set
                    } 
                    std::unique_lock<std::mutex> lock(sleep_mtx);
                    sleeping.store(true,std::memory_order_release);  // Mark the consumer as sleeping
                    PRINT_DBG("Worker thread goes to sleep, waiting for new tasks for function: " + *functionName);                        
                    cv.wait(lock, [&]() { return task.load(std::memory_order_acquire) != nullptr || stop_flag.load(std::memory_order_acquire) == true; });
                    PRINT_DBG("Worker thread woke up, checking for new tasks for function: " + *functionName);
                    sleeping.store(false,std::memory_order_release);  // Mark the consumer as awake                                      
                }
            }
        }

        bool stats_collection;

        std::shared_ptr<std::string> functionName;
        ff_faas_node_t<IN_t, OUT_t>& faas_node;
        std::thread worker_thread;
        // every thread has its own connector
        std::unique_ptr<ff_faas_connector> faasConnector;

        // Task pointer is an atomic to ensure proper synchronization between threads
        alignas(CACHE_LINE_SIZE) std::atomic<typename ff_faas_node_t<IN_t,OUT_t>::internal_task_t*> task;

        // Atomic flag for stopping the consumer
        alignas(CACHE_LINE_SIZE) std::atomic<bool> stop_flag;

        // Atomic flag to track whether the consumer is sleeping or not
        alignas(CACHE_LINE_SIZE) std::atomic<bool> sleeping;

        // Synchronization mechanisms
        alignas(CACHE_LINE_SIZE) std::mutex sleep_mtx;
        alignas(CACHE_LINE_SIZE) std::condition_variable cv;
    };

} // namespace ff

#endif // FF_FAAS_FUNCTION_INVOKER_THREAD_HPP