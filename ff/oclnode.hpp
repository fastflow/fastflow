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
 *  Marco Aldinucci
 *
 *
 */

#ifndef FF_OCLNODE_HPP
#define FF_OCLNODE_HPP

#include <ff/ocl/clEnvironment.hpp>
#include <ff/node.hpp>
#include <vector>

namespace ff{

/*!
 *  \class ff_oclNode
 *  \ingroup buiding_blocks
 *
 *  \brief OpenCL specilisation of ff_node class
 *
 *  Implements the node that is serving a OpenCL device. In general there is one  ff_oclNode
 *  per OpenCL device, even if some expection are under evalutation. Anyway, there is a command
 *  queue per device. Concurrency for accessing the command queue from different  ff_oclNode is
 *  managed by FF.
 *
 */
    
class ff_oclNode : public ff_node {
public:
    //enum device_type { CPU, GPU, ANY};
 
/* cl_device_type - bitfield 
#define CL_DEVICE_TYPE_DEFAULT                      (1 << 0)
#define CL_DEVICE_TYPE_CPU                          (1 << 1)
#define CL_DEVICE_TYPE_GPU                          (1 << 2)
#define CL_DEVICE_TYPE_ACCELERATOR                  (1 << 3)
#define CL_DEVICE_TYPE_CUSTOM                       (1 << 4)
#define CL_DEVICE_TYPE_ALL                          0xFFFFFFFF
*/
                       
    virtual void setDeviceType(cl_device_type dt = CL_DEVICE_TYPE_ALL) { dtype = dt; }
    
    // returns the kind of node
    virtual fftype getFFType() const   { return OCL_WORKER; }

    
    int getOCLID() const { return oclId; }

    // cl_device_type dt =  CL_DEVICE_TYPE_ALL
    std::vector<std::string> getOCLDeviceInfo( ) {
        std::vector<std::string> *res = new std::vector<std::string>();
        for (int j = 0; j < num_devices; j++) {
            char buf[128];
            std::string s1, s2;    
            clGetDeviceInfo(devicelist[j], CL_DEVICE_NAME, 128, buf, NULL);
            // fprintf(stdout, "Device %s supports ", buf);
            s1 = std::string(buf);
            clGetDeviceInfo(devicelist[j], CL_DEVICE_VERSION, 128, buf, NULL);
            //fprintf(stdout, "%s\n", buf);
            s2 = std::string(buf);
            res->push_back(s1+" "+s2);
        }
        return *res;
    }

   
    int getOCL_firstCPUId() const {
        return(getOCL_firstDEVId(CL_DEVICE_TYPE_CPU));
    }

    int getOCL_firstGPUId() const {
        return(getOCL_firstDEVId(CL_DEVICE_TYPE_GPU));
    }
    
    std::vector<int> * getOCL_AllGPUIds() const {
        return (getOCL_DEVIds(CL_DEVICE_TYPE_GPU));
    }
    
protected:
    /**
     * \brief Constructor
     *
     * It construct the OpenCL node for the device.
     *
     */
    ff_oclNode():oclId(-1), deviceId(NULL),devicelist(NULL),num_devices(0) {
        if (devicelist==NULL) inspectOCLDevices();
    };
  
    ~ff_oclNode() {
        if (devicelist!=NULL) free(devicelist);
    }
    
   void inspectOCLDevices () {
        
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, 0, NULL, &num_devices);
        devicelist = (cl_device_id *) malloc(sizeof(cl_device_id) * num_devices);
        clGetDeviceIDs(NULL, CL_DEVICE_TYPE_ALL, num_devices, devicelist, NULL);
    }

    // MA todo - move semantics
    std::vector<int> * getOCL_DEVIds(cl_device_type dt =  CL_DEVICE_TYPE_ALL) const {
        cl_device_type t;
        std::vector<int> *ret = new std::vector<int>();
        for (int j=0; j<num_devices;++j) {
            clGetDeviceInfo(devicelist[j], CL_DEVICE_TYPE, sizeof(t), &t, NULL);
            if (t==dt) {
                ret->push_back(j);
            }
        }
        return ret;
    }
    
     int getOCL_firstDEVId(cl_device_type dt =  CL_DEVICE_TYPE_ALL) const {
        std::vector<int> * list;
        int ret;
        list = getOCL_DEVIds(dt);
        //assert(list->size()>0);
        if (list->size()>0)
            ret = list->at(0);
        else ret = -1;
        delete list;
        return (ret);
    }
   
    
    
   int svc_init() {
       
        if (oclId < 0) oclId = clEnvironment::instance()->getOCLID();

        // initial static mapping
        // A static greedy algorithm is used to allocate openCL components
        switch (dtype) {
        case CL_DEVICE_TYPE_ALL: {
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
        case CL_DEVICE_TYPE_GPU: {
            ssize_t GPUdevId =clEnvironment::instance()->getGPUDevice();
            if( (GPUdevId !=-1) && ( oclId < clEnvironment::instance()->getNumGPU())) { 
                printf("%d: Allocated a GPU device, the id is %ld\n", oclId, GPUdevId);
                deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                return 0;
            }
            printf("%d: cannot allocate a GPU device\n", oclId);            
            return -1;
        } break;
        case CL_DEVICE_TYPE_CPU: {
            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
            if (CPUdevId != -1) {
                printf("%d: Allocated a CPU device (cpuId=%ld)\n",oclId, CPUdevId);
                deviceId=clEnvironment::instance()->getDevice(CPUdevId);        
                return 0;
            }
            printf("%d: cannot allocate a CPU device\n", oclId);
            return -1;
        } break;
        default : std::cerr << "Unknown/not supported device type\n";
            return -1;
        }
        return 0;
    } 

    void svc_end() {}

protected:    
    int            oclId;      // the OpenCL node id
    cl_device_id   deviceId;   // is the id which is provided for user
    cl_device_type    dtype;
    cl_device_id  * devicelist;
    cl_uint num_devices;
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

}
#endif /* FF_OCLNODE_HPP */
