/* -*- Mode: C++; tab-width: 2; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file tpcnode.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow Thread Pool Composer (TPC) interface node
 *
 * This class bridges multicore with FPGAs using the TPC library.
 *
 */

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 2 as published by
 *  the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc., 59
 *  Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this file
 *  does not by itself cause the resulting executable to be covered by the GNU
 *  General Public License.  This exception does not however invalidate any
 *  other reasons why the executable file might be covered by the GNU General
 *  Public License.
 *
 ****************************************************************************
 */

/*  Author: Massimo Torquati torquati@di.unipi.it
 *  - Selptember 2015 first version
 *
 */

#ifndef FF_TPCNODE_HPP
#define FF_TPCNODE_HPP

#include <vector>
#include <algorithm>
#include <ff/node.hpp>

#include <ff/tpc/tpcEnvironment.hpp>
#include <ff/tpcallocator.hpp>
#include <ff/tpc/tpc_api.h>
using namespace rpr;
using namespace tpc;

namespace ff{


namespace internal {
    /**
     *  iovector-like data structure plus memory flags
     */
    struct Arg_t {
        Arg_t():ptr(NULL),size(0),copy(false),reuse(false),release(false) {}
        Arg_t(void *ptr, size_t size, bool copy, bool reuse, bool release):
            ptr(ptr),size(size),copy(copy),reuse(reuse),release(release) {}
        
        void     *ptr;
        size_t    size;
        bool      copy,reuse,release;
    };
} // namespace internal


/**
 * a task to be executed by a TPC based node.
 */
template<typename TaskT_in, typename TaskT_out=TaskT_in>
class baseTPCTask {
public:

    enum class BitFlags {
        COPYTO, 
        DONTCOPYTO, 
        REUSE, 
        DONTREUSE, 
        RELEASE, 
        DONTRELEASE, 
        COPYBACK,
        DONTCOPYBACK
    };

    baseTPCTask():kernel_id(0) {}
    virtual ~baseTPCTask() { }
    
    /* -------------------------------------------- */ 	

    // the user must override this method using:
    // - setInPtr for setting the host-pointer to input data
    // - setOutPtr for setting the host-pointer to output data
    // - other methods from classes derived from baseTPCTask
    // NOTE: order of set*Ptr calls matters! TODO refine interface?
    virtual void setTask(const TaskT_in *t) = 0;
    
    // the user may overrider this method
    // called at the end of the svc. It may be used to 
    // perform per-task host memory cleanup (i.e. releasing the host memory 
    // previously allocated in the setTask function) or to execute a 
    // post-elaboration phase
    virtual TaskT_out *releaseTask(TaskT_in *t) { return t; }

    /* -------------------------------------------- */ 	

    /**
     * set the host-pointer to the input parameter.
     *
     * @param _inPtr the host-pointer
     * @param sizeIn the number of elements in the input array (bytesize=size*sizeof(ptrT))
     * @param copy TODO
     * @param reuse TODO
     * @param release TODO
     */
    template <typename ptrT>
    void setInPtr(const ptrT* _inPtr, size_t size, 
                  const BitFlags copy   =BitFlags::COPYTO, 
                  const BitFlags reuse  =BitFlags::DONTREUSE, 
                  const BitFlags release=BitFlags::DONTRELEASE)  { 
        internal::Arg_t arg(const_cast<ptrT*>(_inPtr),size*sizeof(ptrT),
                            copy==BitFlags::COPYTO,
                            reuse==BitFlags::REUSE,
                            release==BitFlags::RELEASE);
        tpcInput.push_back(arg);
    }

    /**
     * set the host-pointer to the output parameter
     *
     * @see setInPtr()
     * @param copyback TODO
     */
    template <typename ptrT>
    void setOutPtr(const ptrT* _outPtr, size_t size, 
                   const BitFlags copyback =BitFlags::COPYBACK, 
                   const BitFlags reuse    =BitFlags::DONTREUSE, 
                   const BitFlags release  =BitFlags::DONTRELEASE)  { 
        internal::Arg_t arg(const_cast<ptrT*>(_outPtr),size*sizeof(ptrT),
                            copyback==BitFlags::COPYBACK,
                            reuse==BitFlags::REUSE,
                            release==BitFlags::RELEASE);
        tpcOutput.push_back(arg);
    }

    void setKernelId(const tpc_func_id_t id) { kernel_id = id; }

    /* -------------------------------------------- */ 	

    const std::vector<internal::Arg_t> &getInputArgs()  const { return tpcInput; }
    const std::vector<internal::Arg_t> &getOutputArgs() const { return tpcOutput;}
    const tpc_func_id_t       getKernelId()   const { return kernel_id;}

    /* -------------------------------------------- */ 	
    void resetTask() {
        tpcInput.resize(0);
        tpcOutput.resize(0);
    }


protected:
    tpc_func_id_t                kernel_id;
    std::vector<internal::Arg_t> tpcInput;
    std::vector<internal::Arg_t> tpcOutput;
};



/*!
 *  \class ff_tpcNode
 *  \ingroup buiding_blocks
 *
 *  \brief TPC specialisation of the ff_node class
 *
 *  Implements the node that is serving as TPC device.
 *
 * TODO: reuse memory flag management !  ( priorirty very high )
 * TODO: async data transfer to from the device  ( priority medium )
 * TODO : in-place data structures, i.e. outPtr is equal to inPtr !!!! ( priority very high )
 *
 *
 */
template<typename IN_t, typename TPC_t = IN_t, typename OUT_t = IN_t>
class ff_tpcNode_t : public ff_node {
    struct handle_t {
        void         *ptr   = nullptr;
        size_t        size  = 0;
        tpc_handle_t  handle= 0;
    };
    
public:
    typedef IN_t  in_type;
    typedef OUT_t out_type;

    /**
     * \brief Constructor
     *
     * It builds the TPC node for the device.
     *
     */
    ff_tpcNode_t(ff_tpcallocator *alloc = nullptr):
        tpcId(-1),deviceId(-1),my_own_allocator(false),oneshot(false),oneshotTask(nullptr),allocator(alloc) {

        if (allocator == nullptr) {
            my_own_allocator = true;
            allocator = new ff_tpcallocator;
            assert(allocator);
        }

        // TODO: management of multiple TPC devices
    };  
    ff_tpcNode_t(const TPC_t &task, ff_tpcallocator *alloc = nullptr):
        tpcId(-1),deviceId(-1), my_own_allocator(false), oneshot(true),oneshotTask(&task),allocator(alloc) {
        ff_node::skipfirstpop(true);
        Task.setTask(const_cast<TPC_t*>(oneshotTask));

        if (allocator == nullptr) {
            my_own_allocator = true;
            allocator = new ff_tpcallocator;
            assert(allocator);
        }

        // TODO: management of multiple TPC devices
    };  
    virtual ~ff_tpcNode_t()  {
        allocator->releaseAllBuffers(dev_ctx);
        if (my_own_allocator && allocator != nullptr) {
            delete allocator;
            allocator = nullptr;
        }
    }

    virtual int run(bool = false) {
        return ff_node::run();
    }
    
    virtual int wait() {
        return ff_node::wait();
    }

    virtual int run_and_wait_end() {
        if (run() < 0)
            return -1;
        if (wait() < 0)
            return -1;
        return 0;
    }
    
    virtual int run_then_freeze() {
        if (ff_node::isfrozen()) {
            ff_node::thaw(true);
            return 0;
        }
        return ff_node::freeze_and_run();
    }
    virtual int wait_freezing() {
        return ff_node::wait_freezing();
    }
    
    const TPC_t* getTask() const {
        return &Task;
    }
    
    // used to set tasks when in onshot mode
    void setTask(const IN_t &task) {
        Task.resetTask();
        Task.setTask(&task);
    }

    // returns the kind of node
    virtual fftype getFFType() const   { return TPC_WORKER; }

    // TODO currently only one devices can be used
    void setDeviceId(tpc_dev_id_t id)  { deviceId = id; }
    tpc_dev_id_t  getDeviceId()  const  { return deviceId; }    
    int getTPCID() const { return tpcId; }
    
protected:

    int nodeInit() {
        if (tpcId<0) {
            tpcId   = tpcEnvironment::instance()->getTPCID();

            if (deviceId == -1) { // the user didn't set any specific device
                dev_ctx = tpcEnvironment::instance()->getTPCDevice();
            } else 
                abort(); // TODO;

            if (dev_ctx == NULL) return -1;
        }
        return 0;
    }

    int svc_init() { return nodeInit(); }

    // NOTE: this implementation supposes that the number of arguments does not 
    //       change between two different tasks ! Only the size does change.
    void *svc(void *task) {
        if (task) {
            Task.resetTask();
            Task.setTask((IN_t*)task);
        }

        const tpc_func_id_t f_id = Task.getKernelId();
        if (tpc_device_func_instance_count(dev_ctx, f_id) == 0) {
            error("No instances for function with id %d\n", (int)f_id);
            return GO_ON;
        }
        tpc_res_t res;
        tpc_job_id_t j_id = tpc_device_acquire_job_id(dev_ctx, f_id, TPC_ACQUIRE_JOB_ID_BLOCKING);
        
        const std::vector<internal::Arg_t> &inV  = Task.getInputArgs();
        const std::vector<internal::Arg_t> &outV = Task.getOutputArgs();
        
        if (inHandles.size() == 0)  inHandles.resize(inV.size());
        if (outHandles.size() == 0) outHandles.resize(outV.size());


        uint32_t arg_idx = 0;

        for(size_t i=0;i<inV.size();++i) {

            tpc_handle_t handle = inHandles[i].handle;  // previous handle
            if (inHandles[i].size < inV[i].size) {

                if (inHandles[i].handle) { // not first time
                    assert(!inV[i].reuse);
                    allocator->releaseBuffer(inHandles[i].ptr, dev_ctx, inHandles[i].handle);
                }
                if (inV[i].reuse) {
                    handle = allocator->createBufferUnique(inV[i].ptr, dev_ctx, 0, inV[i].size);
                } else {
                    handle = allocator->createBuffer(inV[i].ptr, dev_ctx, 0, inV[i].size);
                }

                if (!handle) {
                    error("ff_tpcNode::svc unable to allocate memory on the TPC device (IN)");
                    inHandles[i].size = 0, inHandles[i].ptr = nullptr, inHandles[i].handle = 0;
                    return (oneshot?NULL:GO_ON);
                }
                inHandles[i].size = inV[i].size, inHandles[i].ptr = inV[i].ptr, inHandles[i].handle = handle;
            }
            
            if (inV[i].copy) {
                res = tpc_device_copy_to(dev_ctx, inV[i].ptr, handle, inV[i].size, TPC_COPY_BLOCKING);
                if (res != TPC_SUCCESS) {
                    error("ff_tpcNode::svc unable to copy data into the device\n");
                    return (oneshot?NULL:GO_ON);
                }
            } 

            res = tpc_device_job_set_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to set argument for the device (IN)\n");
                return (oneshot?NULL:GO_ON);
            }

            ++arg_idx;

        }

        for(size_t i=0;i<outV.size();++i) {
            void *ptr = outV[i].ptr;

            // check if ptr is an in/out parameter
            typename std::vector<handle_t>::iterator it;
            it = std::find_if(inHandles.begin(), inHandles.end(), [ptr](const handle_t &arg) { return arg.ptr == ptr; } );
            
            tpc_handle_t handle;
            if (it == inHandles.end()) { // not found                
                handle = outHandles[i].handle;  // previous handle
                if (outHandles[i].size < outV[i].size) {
                    
                    if (outHandles[i].handle) { // not first time
                        assert(!outV[i].reuse);
                        allocator->releaseBuffer(outHandles[i].ptr, dev_ctx, outHandles[i].handle);
                    }

                    if (outV[i].reuse) {
                        handle = allocator->createBufferUnique(ptr, dev_ctx, 0, outV[i].size);
                    } else {
                        handle = allocator->createBuffer(ptr, dev_ctx, 0, outV[i].size);
                    }
                    
                    if (!handle) {
                        error("ff_tpcNode::svc unable to allocate memory on the TPC device (OUT)");
                        outHandles[i].size = 0, outHandles[i].ptr = nullptr, outHandles[i].handle = 0;
                        return (oneshot?NULL:GO_ON);
                    }
                    outHandles[i].size = outV[i].size, outHandles[i].ptr = ptr, outHandles[i].handle = handle;
                }
            } else handle= (*it).handle; // in/out parameter
           
            res = tpc_device_job_set_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to set argument for the device (OUT)\n");
                return (oneshot?NULL:GO_ON);
            }
            ++arg_idx;
        }
        res = tpc_device_job_launch(dev_ctx, j_id, TPC_JOB_LAUNCH_BLOCKING);
        if (res != TPC_SUCCESS) {
            error("ff_tpcNode::svc error launching kernel on TPC device\n");
            return (oneshot?NULL:GO_ON);
        }
        arg_idx=inV.size();

        
        // TODO: async data transfer
        for(size_t i=0;i<outV.size();++i) {
            tpc_handle_t handle;
            res = tpc_device_job_get_arg(dev_ctx, j_id, arg_idx, sizeof(handle), &handle);
            if (res != TPC_SUCCESS) {
                error("ff_tpcNode::svc unable to get argument handle\n");
                return (oneshot?NULL:GO_ON);
            }
            if (outV[i].copy) {
                res = tpc_device_copy_from(dev_ctx, handle, outV[i].ptr, outV[i].size, TPC_COPY_BLOCKING);
                if (res != TPC_SUCCESS) {
                    error("ff_tpcNode::svc unable to copy data back from the device\n");
                    return (oneshot?NULL:GO_ON);
                }
            } 

            // By default the buffers are not released !
            if (outV[i].release) {
                assert(outHandles[i].handle == handle);
                allocator->releaseBuffer(outHandles[i].ptr, dev_ctx, handle);
                outHandles[i].size = 0, outHandles[i].ptr = nullptr, outHandles[i].handle = 0;
            } 

            ++arg_idx;
        }

        // By default the buffers are not released !
        for(size_t i=0;i<inV.size();++i) {
            if (inV[i].release) {
                allocator->releaseBuffer(inHandles[i].ptr, dev_ctx, inHandles[i].handle);
                inHandles[i].size = 0, inHandles[i].ptr = nullptr, inHandles[i].handle = 0;
            }
        }

        // release job id
        tpc_device_release_job_id(dev_ctx, j_id);

        // per task host memory cleanup phase
        OUT_t * task_out;
        if (task) task_out = Task.releaseTask((IN_t*)task);
        else      task_out = Task.releaseTask(const_cast<IN_t*>(oneshotTask));

        return (oneshot ? NULL : task_out);  
    }
    
protected:    
    int                 tpcId;
    tpc_dev_id_t        deviceId;    
    tpc_dev_ctx_t      *dev_ctx; 

    TPC_t               Task;

    bool                my_own_allocator;
    const bool          oneshot; 
    const TPC_t *const  oneshotTask;
    ff_tpcallocator    *allocator;

    std::vector<handle_t> inHandles;
    std::vector<handle_t> outHandles;
};
    
}
#endif /* FF_TPCNODE_HPP */
