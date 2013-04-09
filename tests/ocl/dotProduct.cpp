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
 * Very basic test for the heterogeneous FastFlow farm.
 * It calculates the dot product for the stream of randomly generated 
 * matrices.
 *
 *    StreamGen ---->  farm(Ocl_worker) ----> StreamColl
 *
 * This is a 3-stage pipeline where the middle stage is a farm havin OpenCL worker.
 *
 */

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>
#include <cstdlib>
#include <fstream>
#include <sys/time.h>

#include <ff/oclnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff;

struct fftask_t{
    fftask_t(size_t size):size(size) {
        inpt1= new cl_float[size];
        inpt2= new cl_float[size];
        outpt= new cl_float[size];
    }

    size_t size;

    cl_float* inpt1;
    cl_float* inpt2;
    cl_float* outpt;    
};

int fillRandom(cl_float * arrayPtr,  const size_t size) {
    if(!arrayPtr) {
        printf("Cannot fill array. NULL pointer.");
        return 0;
    }
    
    /* random initialisation of input */
    for(int i = 0; i < size; i++)
        arrayPtr[i] = (cl_float)(rand()/(RAND_MAX + 1.0)); 
    
    return 1;
}

class StreamGen: public ff_node {
public:
  
    StreamGen(size_t size, size_t strlens):cntr(0), size(size), strlens(strlens){}

    int svc_init() {
        unsigned int   seed = (unsigned int)time(NULL);
        srand(seed);
        return 0;
    }
    
    void * svc(void *) {
        if (cntr<strlens) {
            cntr++;
            fftask_t *t = new fftask_t(size);
            fillRandom(t->inpt1, size);
            fillRandom(t->inpt2, size);
            return (void*)t;
        }
        return NULL;
	}
private:
    size_t cntr;
    size_t size;
    size_t strlens;
};


class StreamColl: public ff_node {
public:
    void *svc(void * t) {
#if defined(CHECK)
        fftask_t *task = (fftask_t*)t;
        size_t size = task->size;
        for(int i=0;i<size;++i) 
            printf("%f ", task->outpt[i]);
        printf("\n");
#endif
        return GO_ON;
    }

};


class Ocl_worker: public ff_oclNode {
public:
    
    Ocl_worker(size_t size): size(size) {}

    void svc_SetUpOclObjects(cl_device_id dId){
        
        // creating the context
        context = clCreateContext(NULL,1,&dId,NULL,NULL,&status);
        printStatus("CreatContext: ", status);
        
        /* create a CL program using the kernel source */
        std::string kernelPath = "./kernels/dot_product.cl";
        std::ifstream t(kernelPath.c_str());
        std::stringstream buffer;
        buffer << t.rdbuf();
        std::string source_string = buffer.str() + '\0';
        const char * source =  source_string.c_str();
        size_t sourceSize[] = { strlen(source) };
        std::cout << sourceSize[0] << " bytes in source" << std::endl;
        program = clCreateProgramWithSource(context,1,(const char **)&source,sourceSize,&status);  
        printStatus("CreateProgramWithSource: ", status); 
        status = clBuildProgram(program,1,&dId,NULL,NULL,NULL);
        size_t len;
        char *buff;
        clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
        buff = new char[len];
        clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, len, buff, NULL);
        
        /*creating command quue*/
        /* The block is to move the declaration of prop closer to its use */
        cl_command_queue_properties prop = 0;
        commandQueue = clCreateCommandQueue(context, dId, prop, &status);
        printStatus("CreateCommandQueue",status);
        
        input1Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_float ) * size, NULL, &status);
        printStatus("createbuf1",status);
        input2Buffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_float ) * size, NULL, &status);
        printStatus("createbuf2",status);
        outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float ) * size, NULL, &status);
        printStatus("createbuf3",status);
        
        /* get a kernel object handle for a kernel with the given name */
        kernel = clCreateKernel(program, "dotProduct", &status);
        printStatus("CreateKernel",status);
        
        /*** Set appropriate arguments to the kernel ***/
        status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&input1Buffer);
        printStatus("setKernel0",status);
        status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&input2Buffer);
        printStatus("setKernel1",status);
        status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&outputBuffer);
        printStatus("setKernel2",status);
    }

    void * svc(void * task) {
        fftask_t *t =(fftask_t*)task;
        assert(t->size == size);

        cl_int   status;
        cl_event events[2];
        size_t globalThreads[1];
        size_t localThreads[1];

        globalThreads[0] = size;
        if (size <256) localThreads[0]  = size;
        else localThreads[0]=256;
        
        // Use clEnqueueWriteBuffer() to write input array A to the device buffer bufferA
        status = clEnqueueWriteBuffer(commandQueue,input1Buffer,CL_FALSE,0,sizeof(cl_float ) * size, t->inpt1,0,NULL,NULL);
        // Use clEnqueueWriteBuffer() to write input array B to the device buffer bufferB
        status = clEnqueueWriteBuffer(commandQueue,input2Buffer,CL_FALSE,0,sizeof(cl_float ) * size, t->inpt2,0,NULL,NULL); 
        
        /* Enqueue a kernel run call.*/
        status = clEnqueueNDRangeKernel(commandQueue,kernel,1,NULL,globalThreads,localThreads,0,NULL,&events[0]);
        
        /* wait for the kernel call to finish execution */
        status = clWaitForEvents(1, &events[0]);
        /* Enqueue readBuffer*/
        status = clEnqueueReadBuffer(commandQueue,outputBuffer,CL_TRUE, 0, size * sizeof(cl_float),t->outpt,0,NULL,&events[1]); 
        
        /* Wait for the read buffer to finish execution */
        status = clWaitForEvents(1, &events[1]);
        status = clReleaseEvent(events[0]);
        status = clReleaseEvent(events[1]); 
        
        return t;
    }
    
    void svc_releaseOclObjects(){
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseCommandQueue(commandQueue);
        clReleaseMemObject(input1Buffer);
        clReleaseMemObject(input2Buffer);
        clReleaseMemObject(outputBuffer);
        clReleaseContext(context);
    }
    
    void  svc_end() {
        std::cout<< "OCL_WORKER : " <<ff_node::get_my_id()<< " has finished the tasks"<<std::endl;
    }
    
private:
    size_t size;
    cl_context context;
    cl_program program;
    cl_command_queue commandQueue;
    cl_mem input1Buffer;
    cl_mem input2Buffer;
    cl_mem outputBuffer;
    cl_kernel kernel;
    cl_int status;
    size_t kernelWorkGroupSize;
};

int main(int argc, char * argv[]) {
    if (argc< 4) {
        printf("use: %s nworkers size streamlen\n", argv[0]);
        return -1;
    }
    int nworkers = atoi(argv[1]);
    size_t size  = atol(argv[2]);
    size_t slen  = atol(argv[3]); 

    ff_pipeline pipe;
    StreamGen streamgen(size,slen);
    pipe.add_stage(&streamgen);

    ff_farm<> farm;
    //add collector
    farm.add_collector(NULL);
    // add workers
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) 
        w.push_back(new Ocl_worker(size));
    farm.add_workers(w);

    pipe.add_stage(&farm);

    StreamColl streamcoll;
    pipe.add_stage(&streamcoll);

    // running the farm
    ffTime(START_TIME);
    if (pipe.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }     
    ffTime(STOP_TIME);
    
    std::cerr << "DONE, pipe time= "  << pipe.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    return 0;
}

