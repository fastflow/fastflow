/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file map.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief OpenCL map pattern
 *
 */

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
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

 
#ifndef FF_MAP_OCL_HPP
#define FF_MAP_OCL_HPP


#include <string>
#include <fstream>
#include <ff/oclnode.hpp>


namespace ff {

// map base task for OpenCL implementation
template<typename Tin_, typename Tout_ = Tin_>
class baseOCLTask {
public:
    typedef Tin_ Tin;
    typedef Tout_ Tout;

    baseOCLTask(): inPtr(NULL),outPtr(NULL),size_in(0),size_out(0),reduceVar(NULL) { }   
    virtual ~baseOCLTask() { }
    
    // user must override this method
    virtual void setTask(void* t) = 0;

    size_t getBytesizeIn()   const { return getSizeIn() * sizeof(Tin); }
    size_t getBytesizeOut()  const { return getSizeOut() * sizeof(Tout); }

    void setSizeIn(size_t   sizeIn)   {	size_in   = sizeIn;   }
    void setSizeOut(size_t  sizeOut)  {	size_out  = sizeOut;   }

    size_t getSizeIn()   const { return size_in;   }
    size_t getSizeOut()  const { return (size_out==0)?size_in:size_out;  }

    void setInPtr(Tin*     _inPtr)     { inPtr   = _inPtr;  }
    void setOutPtr(Tout*   _outPtr)    { outPtr  = _outPtr;  }

    Tin*   getInPtr()   const { return inPtr;   }
    Tout*  getOutPtr()  const { return outPtr;  }
    
    void setReduceVar(Tout *r) { reduceVar = r;  }
    Tout *getReduceVar() const { return reduceVar;  }

protected:
    Tin    *inPtr;
    Tout   *outPtr;
    size_t  size_in, size_out;
    Tout   *reduceVar;
};



    
    /* The following OpenCL code macros have been derived from the SkePU OpenCL code
     * http://www.ida.liu.se/~chrke/skepu/
     *
     */
#define FFMAPFUNC(name, basictype, param, code)                         \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(" #basictype " " #param ") {" #code ";}\n"      \
"__kernel void kern_" #name "(__global " #basictype "* input,\n"        \
"                             __global " #basictype "* output,\n"       \
"                             const uint maxItems) {\n"                 \
"           int i = get_global_id(0);\n"                                \
"           uint gridSize = get_local_size(0)*get_num_groups(0);\n"     \
"           while(i < maxItems)  {\n"                                   \
"              output[i] = f" #name "(input[i]);\n"                     \
"              i += gridSize;\n"                                        \
"           }\n"                                                        \
"}"

#define FFMAPFUNC2(name, outT, inT, param, code)                        \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #outT "|"                                                       \
#outT " f" #name "(" #inT " " #param ") {" #code ";}\n"                 \
"__kernel void kern_" #name "(__global " #inT  "* input,\n"             \
"                             __global " #outT "* output,\n"            \
"                             const uint maxItems) {\n"                 \
"           int i = get_global_id(0);\n"                                \
"           uint gridSize = get_local_size(0)*get_num_groups(0);\n"     \
"           while(i < maxItems)  {\n"                                   \
"              output[i] = f" #name "(input[i]);\n"                     \
"              i += gridSize;\n"                                        \
"           }\n"                                                        \
"}"



#define FFREDUCEFUNC(name, basictype, param1, param2, code)             \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(" #basictype " " #param1 ", " #basictype " " #param2 ") {" #code ";}\n"    \
"__kernel void kern_" #name "(__global " #basictype "* input, __global " #basictype "* output, const uint n, __local " #basictype "* sdata) {\n" \
"        uint blockSize = get_local_size(0);\n"                         \
"        uint tid = get_local_id(0);\n"                                 \
"        uint i = get_group_id(0)*blockSize + get_local_id(0);\n"       \
"        uint gridSize = blockSize*get_num_groups(0);\n"                \
"        float result = 0;\n"                                           \
"        if(i < n) { result = input[i]; i += gridSize; }\n"             \
"        while(i < n) {\n"                                              \
"          result = f" #name "(result, input[i]);\n"                    \
"          i += gridSize;\n"                                            \
"        }\n"                                                           \
"        sdata[tid] = result;\n"                                        \
"        barrier(CLK_LOCAL_MEM_FENCE);\n"                               \
"        if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(tid == 0) output[get_group_id(0)] = sdata[tid];\n"          \
"}\n"
    


    // FIX: da rifare tutte !!

#define NEWMAP(name, task_t, f, input, sz)               \
    ff_mapOCL<task_t > *name =                           \
        new ff_mapOCL<task_t>(f, input,sz)
#define NEWMAPONSTREAM(task_t, f)                        \
    new ff_mapOCL<task_t>(f)
#define DELETEMAP(name)                                  \
    delete name
#define NEWREDUCE(name, task_t, f, input, sz)            \
    ff_reduceOCL<task_t> *name =                         \
        new ff_reduceOCL<task_t>(f, input,sz)
#define NEWMAPREDUCE(name, task_t, mapf, redf, input, sz)\
    ff_mapreduceOCL<task_t> *name =                      \
        new ff_mapreduceOCL<task_t>(mapf, redf, input,sz)
#define NEWREDUCEONSTREAM(task_t, f)                     \
    new ff_reduceOCL<task_t>(f)
#define DELETEREDUCE(name)                               \
    delete name

/*
 * \class ff_ocl
 *  \ingroup high_level_patterns
 *
 * \brief The FF ocl kernel interface 
 *
 * The kernel skeleton using OpenCL
 *
 * This class is defined in \ref map.hpp
 * 
 */
template<typename T>
class ff_ocl: public ff_oclNode {
private:
    void setcode(const std::string &codestr1, const std::string &codestr2) {
        int n = codestr1.find("|");
        assert(n>0);
        ff_ocl<T>::kernel_name1 = codestr1.substr(0,n);
        const std::string &tmpstr = codestr1.substr(n+1);
        n = tmpstr.find("|");
        assert(n>0);
        
        // checking for double type
        if (tmpstr.substr(0,n) == "double") {
            kernel_code = "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n" +
                tmpstr.substr(n+1);
        } else 
            kernel_code  = tmpstr.substr(n+1);        
        
        // checking for extra code needed to compile the kernels                
        std::ifstream ifs(FF_OPENCL_DATATYPES_FILE);
        if (ifs.is_open())
            kernel_code.insert( kernel_code.begin(), std::istreambuf_iterator<char>(ifs), std::istreambuf_iterator<char>());

        if (codestr2 != "") {
            n = codestr2.find("|");
            assert(n>0);
            ff_ocl<T>::kernel_name2 += codestr2.substr(0,n);
            const std::string &tmpstr = codestr2.substr(n+1);
            n = tmpstr.find("|");
            assert(n>0);
        
            // checking for double type
            if (tmpstr.substr(0,n) == "double") {
                kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n" +
                    tmpstr.substr(n+1);
            } else 
                kernel_code  += tmpstr.substr(n+1);        
        }
    }
public:

    ff_ocl(const std::string &codestr1, 
           const std::string &codestr2=std::string("")):
        oneshot(false) {
        setcode(codestr1, codestr2);
        oldSize = 0;
        oldOutPtr = false;
    }
    ff_ocl(const T &task, 
           const std::string &codestr1, 
           const std::string &codestr2=std::string("")):
        oneshot(true) {
        setcode(codestr1,codestr2);
        Task.setTask((void*)&task);
        oldSize = 0;
        oldOutPtr = false;
    }


    void setKernel2() {
        clReleaseKernel(kernel);
        kernel = clCreateKernel(program, kernel_name2.c_str(), &status);
        checkResult(status, "CreateKernel (2)");
        
        status = clGetKernelWorkGroupInfo(kernel, baseclass_ocl_node_deviceId,
                                  CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t), &workgroup_size,0);
        checkResult(status, "GetKernelWorkGroupInfo");
    }

    const T* getTask() const { return &Task; }
protected:
    inline void checkResult(cl_int s, const char* msg) {
        if(s != CL_SUCCESS) {
            std::cerr << msg << ":";
            printOCLErrorString(s,std::cerr);
        }
    }    

    void svc_SetUpOclObjects(cl_device_id dId) {	                        
        cl_int status;							
        context = clCreateContext(NULL,1,&dId,NULL,NULL,&status);		
        checkResult(status, "creating context");				
        
        cmd_queue = clCreateCommandQueue (context, dId, 0, &status);	
        checkResult(status, "creating command-queue");			
        
        size_t sourceSize = kernel_code.length();        
        const char* code = kernel_code.c_str();

        //printf("code=\n%s\n", code);
        program = clCreateProgramWithSource(context,1, &code, &sourceSize,&status);    
        checkResult(status, "creating program with source");		        

        status = clBuildProgram(program,1,&dId,NULL,NULL,NULL);		
        checkResult(status, "building program");				

#if 0
        { // DEBUGGING CODE for checking OCL compilation errors
            printf("\nFail to build the program\n");
            size_t len;
            clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
            printf("LOG len %ld\n", len);
            char *buffer = (char*)calloc(len, sizeof(char)); assert(buffer);
            clGetProgramBuildInfo(program,dId,CL_PROGRAM_BUILD_LOG,len*sizeof(char),buffer,NULL);
            printf("LOG: %s\n\n", buffer);
        }
#endif

        kernel = clCreateKernel(program, kernel_name1.c_str(), &status);			
        checkResult(status, "CreateKernel");				
        
        status = clGetKernelWorkGroupInfo(kernel, dId,			
                                          CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t), &workgroup_size,0); 
        checkResult(status, "GetKernelWorkGroupInfo");			

        // allocate memory on device having the initial size
        if (Task.getBytesizeIn()>0) {
            inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,		
                                         Task.getBytesizeIn(), NULL, &status);          
            checkResult(status, "CreateBuffer input (1)");	
            oldSize = Task.getBytesizeIn();
        }
    }									

    void svc_releaseOclObjects(){						
        clReleaseKernel(kernel);						
        clReleaseProgram(program);						
        clReleaseCommandQueue(cmd_queue);					
        clReleaseMemObject(inputBuffer);
        if (oldOutPtr)
            clReleaseMemObject(outputBuffer);					
        clReleaseContext(context);						
    }									

    cl_mem* getInputBuffer()  const { return (cl_mem*)&inputBuffer;}
    cl_mem* getOutputBuffer() const { return (cl_mem*)&outputBuffer;}
    void swapInOut(){inputBuffer=outputBuffer;}
    
protected:
    const bool oneshot;
    T Task;
    std::string kernel_code;
    std::string kernel_name1;
    std::string kernel_name2;
    size_t workgroup_size;
    size_t oldSize;
    bool   oldOutPtr;
    cl_context context;							
    cl_program program;							
    cl_command_queue cmd_queue;						
    cl_mem inputBuffer;							
    cl_mem outputBuffer;							
    cl_kernel kernel;
    cl_int status;							
};



/*
 * \class ff_mapOCL
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The map skeleton.
 *
 * The map skeleton using OpenCL
 *
 * This class is defined in \ref map.hpp
 * 
 */
template<typename T>
class ff_mapOCL: public ff_ocl<T> {
public:
    
    ff_mapOCL(std::string codestr):ff_ocl<T>(codestr) { 
    }    
    ff_mapOCL(const T &task, std::string codestr):
        ff_ocl<T>(task, codestr) { 
        ff_node::skipfirstpop(true);
    }    
    
    int  run(bool=false) { return  ff_node::run(); }    
    int  wait() { return ff_node::wait(); }    

    int run_and_wait_end() {
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }

    double ffTime()  { return ff_node::ffTime();  }
    double ffwTime() { return ff_node::wffTime(); }

    const T* getTask() const { return ff_ocl<T>::getTask(); }
    
protected:
    
    void * svc(void* task) {				                        
        cl_int   status;
        cl_event events[2];
        size_t globalThreads[1];
        size_t localThreads[1];
        
        if (task) ff_ocl<T>::Task.setTask(task);
        const size_t size = ff_ocl<T>::Task.getSizeIn();
        void* inPtr  = ff_ocl<T>::Task.getInPtr();
        void* outPtr = ff_ocl<T>::Task.getOutPtr();
        assert(outPtr != NULL);

        if (size  < ff_ocl<T>::workgroup_size) {					
            localThreads[0]  = size;					
            globalThreads[0] = size;					
        } else {								
            localThreads[0]  = ff_ocl<T>::workgroup_size;					
            globalThreads[0] = nextMultipleOfIf(size,ff_ocl<T>::workgroup_size);	
        }									

        if ( ff_ocl<T>::oldSize < ff_ocl<T>::Task.getBytesizeIn() ) {
            if (ff_ocl<T>::oldSize != 0) clReleaseMemObject(ff_ocl<T>::inputBuffer);
            ff_ocl<T>::inputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,		
                                                    ff_ocl<T>::Task.getBytesizeIn(), NULL, &status); 
            ff_ocl<T>::checkResult(status, "CreateBuffer input (2)");	
            ff_ocl<T>::oldSize = ff_ocl<T>::Task.getBytesizeIn();
        }
        
        if (inPtr == outPtr) {                       
            ff_ocl<T>::outputBuffer = ff_ocl<T>::inputBuffer;
        } else { 

            // FIX:  controllare se e' possibile evitare di allocare memoria sul device
            //       se non serve

            if (ff_ocl<T>::oldOutPtr) clReleaseMemObject(ff_ocl<T>::outputBuffer);
            ff_ocl<T>::outputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,		
                                                     ff_ocl<T>::Task.getBytesizeOut(), NULL, &status);              
            ff_ocl<T>::checkResult(status, "CreateBuffer output");	
            ff_ocl<T>::oldOutPtr = true;
        }
        
        status = clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem), ff_ocl<T>::getInputBuffer());
        ff_ocl<T>::checkResult(status, "setKernelArg input");				
        status = clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem), ff_ocl<T>::getOutputBuffer());
        ff_ocl<T>::checkResult(status, "setKernelArg output");				
        status = clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint), (void *)&size);				
        ff_ocl<T>::checkResult(status, "setKernelArg size");				
        
        status = clEnqueueWriteBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::inputBuffer,CL_FALSE,0,	
                                      ff_ocl<T>::Task.getBytesizeIn(), ff_ocl<T>::Task.getInPtr(),
                                      0,NULL,NULL);			
        ff_ocl<T>::checkResult(status, "copying Task to device input-buffer");		

        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,		
                                        globalThreads, localThreads,0,NULL,&events[0]);             

        status |= clWaitForEvents(1, &events[0]);
        status |= clEnqueueReadBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::outputBuffer,CL_TRUE, 0,	
                                      ff_ocl<T>::Task.getBytesizeOut(), outPtr,0,NULL,&events[1]);     
        status |= clWaitForEvents(1, &events[1]);				

        clReleaseEvent(events[0]);					
        clReleaseEvent(events[1]);					
         
        //return (ff_ocl<T>::oneshot?NULL:task);
        return (ff_ocl<T>::oneshot?NULL:task);
    }   
};



/*
 * \class ff_reduceOCL
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The reduce skeleton using OpenCL
 *
 * 
 */
template<typename T>
class ff_reduceOCL: public ff_ocl<T> {
public:
    typedef typename T::Tout Tout;

    ff_reduceOCL(std::string codestr):
        ff_ocl<T>(codestr) {}    

    ff_reduceOCL(const T &task, std::string codestr):
        ff_ocl<T>(task, codestr) {
        ff_node::skipfirstpop(true);
    }    
    
    int  run(bool=false) { return  ff_node::run(); }    
    int  wait() { return ff_node::wait(); }    

    int run_and_wait_end() {
        if (run()<0) return -1;           
        if (wait()<0) return -1;
        return 0;
    }

    double ffTime()  { return ff_node::ffTime();  }
    double ffwTime() { return ff_node::wffTime(); }

    const T* getTask() const { return ff_ocl<T>::getTask(); }

protected:
    inline void checkResult(cl_int s, const char* msg) {
        if(s != CL_SUCCESS) {
            std::cerr << msg << ":";
            printOCLErrorString(s,std::cerr);
        }
    }

    /*!
     * Computes the number of threads and blocks to use for the reduction kernel.
     */
    inline void getBlocksAndThreads(const size_t size, 
                                    const size_t maxBlocks, const size_t maxThreads, 
                                    size_t & blocks, size_t &threads) {

        threads = (size < maxThreads*2) ? nextPowerOf2((size + 1)/ 2) : maxThreads;
        blocks  = (size + (threads * 2 - 1)) / (threads * 2);
        blocks  = std::min(maxBlocks, blocks);
    }

    void * svc(void* task) {				                        
        cl_int   status = CL_SUCCESS;
        cl_event events[2];
        size_t globalThreads[1];
        size_t localThreads[1];
        
        if (task) ff_ocl<T>::Task.setTask(task);
        size_t size       = ff_ocl<T>::Task.getSizeIn();
        size_t elemSize   = sizeof(Tout);
        size_t numBlocks  = 0;
        size_t numThreads = 0;

        /* 64 and 256 are the max number of blocks and threads we want to use */
        getBlocksAndThreads(size, 64, 256, numBlocks, numThreads);

        size_t outMemSize = 
            (numThreads <= 32) ? (2 * numThreads * elemSize) : (numThreads * elemSize); 

        localThreads[0]  = numThreads;
        globalThreads[0] = numBlocks * numThreads;

        ff_ocl<T>::inputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_ONLY,ff_ocl<T>::Task.getBytesizeIn(), NULL, &status);     
        ff_ocl<T>::checkResult(status, "CreateBuffer input (3)");

        ff_ocl<T>::outputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,numBlocks*elemSize, NULL, &status);
        ff_ocl<T>::checkResult(status, "CreateBuffer output");					
        
        status |= clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem), ff_ocl<T>::getInputBuffer());			
        status |= clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem), ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint), (void *)&size);				
        status != clSetKernelArg(ff_ocl<T>::kernel, 3, outMemSize, NULL);
        checkResult(status, "setKernelArg ");				
        
        status = clEnqueueWriteBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::inputBuffer,CL_FALSE,0,	
                                      ff_ocl<T>::Task.getBytesizeIn(), ff_ocl<T>::Task.getInPtr(),
                                      0,NULL,NULL);			
        ff_ocl<T>::checkResult(status, "copying Task to device input-buffer");		
        
        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,		
                                        globalThreads,localThreads,0,NULL,&events[0]);             
        status = clWaitForEvents(1, &events[0]);				

        // Sets the kernel arguments for second reduction
        size = numBlocks;
        status |= clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem),ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem),ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint),(void*)&size);
        status |= clSetKernelArg(ff_ocl<T>::kernel, 3, outMemSize, NULL);
        ff_ocl<T>::checkResult(status, "setKernelArg ");			

        localThreads[0]  = numThreads;
        globalThreads[0] = numThreads;

        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,		
                                        globalThreads,localThreads,0,NULL,&events[0]); 
        Tout *reduceVar = ff_ocl<T>::Task.getReduceVar();
        assert(reduceVar != NULL);
        status |= clWaitForEvents(1, &events[0]);
        status |= clEnqueueReadBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::outputBuffer,CL_TRUE, 0,	
                                      elemSize, reduceVar ,0,NULL,&events[1]);          
        status |= clWaitForEvents(1, &events[1]);
        ff_ocl<T>::checkResult(status, "ERROR during OpenCL computation");		

        clReleaseEvent(events[0]);					
        clReleaseEvent(events[1]);					
        
        //return (ff_ocl<T>::oneshot?NULL:task);
        return (ff_ocl<T>::oneshot?NULL:task);
    }									
};

/*
 * \class ff_mapreduceOCL
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The map reduce skeleton using OpenCL
 *
 *
 */
template<typename T>
class ff_mapreduceOCL: public ff_ocl<T> {
public:
    typedef typename T::Tout Tout;

	ff_mapreduceOCL(std::string mapf, std::string reducef):ff_ocl<T>(mapf,reducef) {}
	ff_mapreduceOCL(const T &task, std::string mapf, std::string reducef):
        ff_ocl<T>(task, mapf, reducef) {
        ff_node::skipfirstpop(true);
    }

    int  run(bool=false) { return  ff_node::run(); }
    int  wait() { return ff_node::wait(); }

    int run_and_wait_end() {
        if (run()<0) return -1;
        if (wait()<0) return -1;
        return 0;
    }

    double ffTime()  { return ff_node::ffTime();  }
    double ffwTime() { return ff_node::wffTime(); }

    const T* getTask() const { return ff_ocl<T>::getTask(); }

protected:
    inline void checkResult(cl_int s, const char* msg) {
        if(s != CL_SUCCESS) {
            std::cerr << msg << ":";
            printOCLErrorString(s,std::cerr);
        }
    }

    /*!
     * Computes the number of threads and blocks to use for the reduction kernel.
     */
    inline void getBlocksAndThreads(const size_t size,
                                    const size_t maxBlocks, const size_t maxThreads,
                                    size_t & blocks, size_t &threads) {

        threads = (size < maxThreads*2) ? nextPowerOf2((size + 1)/ 2) : maxThreads;
        blocks  = (size + (threads * 2 - 1)) / (threads * 2);
        blocks  = std::min(maxBlocks, blocks);
    }

    void * svc(void* task) {
        cl_int   status = CL_SUCCESS;
        cl_event events[2];
        size_t globalThreads[1];
        size_t localThreads[1];

        if (task) ff_ocl<T>::Task.setTask(task);
        size_t size       = ff_ocl<T>::Task.getSizeIn();
        void* inPtr  = ff_ocl<T>::Task.getInPtr();
        void* outPtr = ff_ocl<T>::Task.getOutPtr();

        // MAP
        if (size  < ff_ocl<T>::workgroup_size) {
        	localThreads[0]  = size;
        	globalThreads[0] = size;
        } else {
        	localThreads[0]  = ff_ocl<T>::workgroup_size;
        	globalThreads[0] = nextMultipleOfIf(size,ff_ocl<T>::workgroup_size);
        }

        if ( ff_ocl<T>::oldSize < ff_ocl<T>::Task.getBytesizeIn() ) {
        	if (ff_ocl<T>::oldSize != 0) clReleaseMemObject(ff_ocl<T>::inputBuffer);
        	ff_ocl<T>::inputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,
                                      ff_ocl<T>::Task.getBytesizeIn(), NULL, &status);
        	ff_ocl<T>::checkResult(status, "CreateBuffer input (2)");
        	ff_ocl<T>::oldSize = ff_ocl<T>::Task.getBytesizeIn();
        }

        if ( inPtr == outPtr ) {
        	ff_ocl<T>::outputBuffer = ff_ocl<T>::inputBuffer;
        } else {
        	if (ff_ocl<T>::oldOutPtr) clReleaseMemObject(ff_ocl<T>::outputBuffer);
        	ff_ocl<T>::outputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,
                                          ff_ocl<T>::Task.getBytesizeOut(), NULL, &status);
        	ff_ocl<T>::checkResult(status, "CreateBuffer output");
        	ff_ocl<T>::oldOutPtr = true;
        }

        status = clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem), ff_ocl<T>::getInputBuffer());
        ff_ocl<T>::checkResult(status, "setKernelArg input");
        status = clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem), ff_ocl<T>::getOutputBuffer());
        ff_ocl<T>::checkResult(status, "setKernelArg output");
        status = clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint), (void *)&size);
        ff_ocl<T>::checkResult(status, "setKernelArg size");

        status = clEnqueueWriteBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::inputBuffer,CL_FALSE,0,
                                      ff_ocl<T>::Task.getBytesizeIn(), ff_ocl<T>::Task.getInPtr(),
        		0,NULL,NULL);
        ff_ocl<T>::checkResult(status, "copying Task to device input-buffer");

        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,
        		globalThreads, localThreads,0,NULL,&events[0]);

        status |= clWaitForEvents(1, &events[0]);

        // REDUCE
        ff_ocl<T>::setKernel2();

        size_t elemSize   = sizeof(typename T::Tout);
        size_t numBlocks  = 0;
        size_t numThreads = 0;

        // 64 and 256 are the max number of blocks and threads we want to use
        getBlocksAndThreads(size, 64, 256, numBlocks, numThreads);

        size_t outMemSize =
            (numThreads <= 32) ? (2 * numThreads * elemSize) : (numThreads * elemSize);

        localThreads[0]  = numThreads;
        globalThreads[0] = numBlocks * numThreads;

        // input buffer is the output buffer from map stage
        ff_ocl<T>::swapInOut(); 

        ff_ocl<T>::outputBuffer = clCreateBuffer(ff_ocl<T>::context, CL_MEM_READ_WRITE,numBlocks*elemSize, NULL, &status);
        ff_ocl<T>::checkResult(status, "CreateBuffer output");

        status |= clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem), ff_ocl<T>::getInputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem), ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint), (void *)&size);
        status != clSetKernelArg(ff_ocl<T>::kernel, 3, outMemSize, NULL);
        checkResult(status, "setKernelArg ");

        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,
                                        globalThreads,localThreads,0,NULL,&events[0]);
        status = clWaitForEvents(1, &events[0]);

        // Sets the kernel arguments for second reduction
        size = numBlocks;
        status |= clSetKernelArg(ff_ocl<T>::kernel, 0, sizeof(cl_mem),ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 1, sizeof(cl_mem),ff_ocl<T>::getOutputBuffer());
        status |= clSetKernelArg(ff_ocl<T>::kernel, 2, sizeof(cl_uint),(void*)&size);
        status |= clSetKernelArg(ff_ocl<T>::kernel, 3, outMemSize, NULL);
        ff_ocl<T>::checkResult(status, "setKernelArg ");

        localThreads[0]  = numThreads;
        globalThreads[0] = numThreads;

        status = clEnqueueNDRangeKernel(ff_ocl<T>::cmd_queue,ff_ocl<T>::kernel,1,NULL,
                                        globalThreads,localThreads,0,NULL,&events[0]);
        Tout *reduceVar = ff_ocl<T>::Task.getReduceVar();
        assert(reduceVar != NULL);
        status |= clWaitForEvents(1, &events[0]);
        status |= clEnqueueReadBuffer(ff_ocl<T>::cmd_queue,ff_ocl<T>::outputBuffer,CL_TRUE, 0,
                                      elemSize, reduceVar ,0,NULL,&events[1]);
        status |= clWaitForEvents(1, &events[1]);
        ff_ocl<T>::checkResult(status, "ERROR during OpenCL computation");

        clReleaseEvent(events[0]);
        clReleaseEvent(events[1]);

        //return (ff_ocl<T>::oneshot?NULL:task);
        return (ff_ocl<T>::oneshot?NULL:task);
    }
};

} // namespace ff

#endif /* FF_MAP_OCL_HPP */

