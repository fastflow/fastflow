/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/* 
 * Author: Massimo Torquati, Marco Aldinucci
 * Date:   August 2015
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <string>
#include <vector>
#include <iostream>
#include <ff/ocl/clEnvironment.hpp>

using namespace ff;

int main(int argc, char * argv[]) {
    std::vector<std::string> res = clEnvironment::instance()->getDevicesInfo();
    
    for (size_t i=0; i<res.size(); ++i)
        std::cout << i << " - " << res[i] << std::endl;
    
    std::cout << "First CPU is " << clEnvironment::instance()->getCPUDevice() << "\n";
    std::cout << "First GPU is " << clEnvironment::instance()->getGPUDeviceRR() << "\n";

    std::vector<ssize_t> allgpus = clEnvironment::instance()->getAllGPUDevices();
    for (size_t i=0; i<allgpus.size(); ++i)
        std::cout << "GPU #" << i << " ID " << allgpus.at(i) << std::endl;
    
    return 0;
}
