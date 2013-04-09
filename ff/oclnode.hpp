/* -*- Mode: C++; tab-width: 2; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file oclnode.hpp
 * \ingroup streaming_network_arbitrary_shared_memory
 *
 * \brief TODO
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
  Mehdi Goli: m.goli@rgu.ac.uk*/

#ifndef _FF_OCLNODE_HPP_
#define _FF_OCLNODE_HPP_

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
 * \ingroup streaming_network_arbitrary_shared_memory
 *
 * \brief TODO
 *
 * This class is defined in \ref ff_oclnode.hpp
 *
 */
class ff_oclNode : public ff_node, public Ocl_Utilities { 
public:

    /**
     * TODO
     */
    virtual void  svc_SetUpOclObjects(cl_device_id id)=0;
    
    /**
     * TODO
     */
    virtual void  svc_releaseOclObjects()=0;
     
protected:    
   
    /**
     * TODO
     */
    bool initOCLInstance() {
        Environment::instance()->createEntry(tId, this);
        if(tId>=0) {
            baseclass_ocl_node_deviceId = Environment::instance()->getDeviceId(tId);
            return true;
        }
        return false;
    }
    
    /**
     * TODO
     */
    ff_oclNode():baseclass_ocl_node_deviceId(NULL),tId(-1){ }
    
    /**
     * TODO
     */
    bool  device_rules(cl_device_id id){ return true;}
    
    /**
     * TODO
     */
    inline void svc_createOCL(){
        if (baseclass_ocl_node_deviceId==NULL) {
            if (initOCLInstance()){
                svc_SetUpOclObjects(baseclass_ocl_node_deviceId);
            }else{
                error("ERROR: Instantiating the Device: Failed to instatiate the device!\n");
                abort();
            }
        }
        else if (evaluation()) {
            svc_releaseOclObjects();
            svc_SetUpOclObjects(baseclass_ocl_node_deviceId);
        }
    } 
    
    /**
     * TODO
     */
    inline void svc_releaseOCL(){ 
        svc_releaseOclObjects();        
    }
    
    /**
     * TODO
     */
    inline bool evaluation(){
        cl_device_id nDId = Environment::instance()->reallocation(tId); 
        if (nDId != baseclass_ocl_node_deviceId) {
            baseclass_ocl_node_deviceId = nDId; 
            return true;
        }
        return false;
    }
    
private:
    int tId; // the node id
    cl_device_id baseclass_ocl_node_deviceId; // is the id which is provided for user
};

/*!
 * @}
 * \endlink
 */
}
#endif
