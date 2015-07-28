/* -*- Mode: C++; tab-width: 2; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file oclnode.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow OpenCL interface node
 *
 * @detail This class bridges multicore with GPGPUs using OpenCL
 *
 * \note This class is deprecated in the current form. It will be updated soon  
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

/*  Mehdi Goli:            m.goli@rgu.ac.uk    goli.mehdi@gmail.com
 *  Massimo Torquati:      torquati@di.unipi.it
 *
 *
 */

#ifndef FF_OCLNODE_HPP
#define FF_OCLNODE_HPP

#include <ff/ocl/clEnvironment.hpp>
#include <ff/node.hpp>

namespace ff{

/*!
 * \ingroup streaming_network_arbitrary_shared_memory
 *
 * @{
 */

/*!
 * \class ff_oclNode
 * \ingroup building_blocks
 *
 * \brief OpenCL implementation of FastFlow node
 *
 *
 */
class ff_oclNode : public ff_node {
public:
    enum device_type { CPU, GPU, ANY};

    virtual void setDeviceType(device_type dt = device_type::ANY) { dtype = dt; }
    
    // returns the kind of node
    virtual fftype getFFType() const   { return OCL_WORKER; }

    int getOCLID() const { return oclId; }
     
protected:
    /**
     * \brief Constructor
     *
     * It construct the OpenCL node for the device.
     *
     */
    ff_oclNode():oclId(-1), deviceId(NULL) {}
    
    
    int svc_init() {
        if (oclId < 0) oclId = clEnvironment::instance()->getOCLID();

        // initial static mapping
        // A static greedy algorithm is used to allocate openCL components
        switch (dtype) {
        case ANY: {
            ssize_t GPUdevId =clEnvironment::instance()->getGPUDevice();
            if( (GPUdevId !=-1) && ( oclId < clEnvironment::instance()->getNumGPU())) { 
                printf("%d: Allocated a GPU device, the id is %ld\n", oclId, GPUdevId);
                deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                return 0;
            }
            // fall back to CPU either GPU has reached its max or there is no GPU available
            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
            if (CPUdevId != -1) {
                printf("%d: Allocated a CPU device as either no GPU device is available or no GPU slot is available (cpuId=%ld)\n",oclId, CPUdevId);
                deviceId=clEnvironment::instance()->getDevice(CPUdevId);
                return 0;
            }
            printf("%d: cannot allocate neither a GPU nor a CPU device\n", oclId);            
            return -1;
        } break;
        case GPU: {
            ssize_t GPUdevId =clEnvironment::instance()->getGPUDevice();
            if( (GPUdevId !=-1) && ( oclId < clEnvironment::instance()->getNumGPU())) { 
                printf("%d: Allocated a GPU device, the id is %ld\n", oclId, GPUdevId);
                deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                return 0;
            }
            printf("%d: cannot allocate a GPU device\n", oclId);            
            return -1;
        } break;
        case CPU: {
            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
            if (CPUdevId != -1) {
                printf("%d: Allocated a CPU device (cpuId=%ld)\n",oclId, CPUdevId);
                deviceId=clEnvironment::instance()->getDevice(CPUdevId);        
                return 0;
            }
            printf("%d: cannot allocate a CPU device\n", oclId);
            return -1;
        } break;
        default : return -1;
        }
        return 0;
    } 

    void svc_end() {}

protected:    
    int            oclId;      // the OpenCL node id
    cl_device_id   deviceId;   // is the id which is provided for user
    device_type    dtype; 
};

template<typename IN, typename OUT=IN>
struct ff_oclNode_t: ff_oclNode {
    typedef IN  in_type;
    typedef OUT out_type;
    ff_oclNode_t():
        GO_ON((OUT*)FF_GO_ON),
        EOS((OUT*)FF_EOS),
        GO_OUT((OUT*)FF_GO_OUT),
        EOS_NOFREEZE((OUT*)FF_EOS_NOFREEZE) {}
    OUT *GO_ON, *EOS, *GO_OUT, *EOS_NOFREEZE;
    virtual ~ff_oclNode_t()  {}
    virtual OUT* svc(IN*)=0;
    inline  void *svc(void *task) { return svc(reinterpret_cast<IN*>(task));};
};

/*!
 * @}
 * \endlink
 */
}
#endif /* FF_OCLNODE_HPP */
