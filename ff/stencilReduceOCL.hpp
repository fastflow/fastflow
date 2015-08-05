/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file stencilReduceOCL.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief StencilReduceLoop data-parallel pattern and derived data-parallel patterns
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

/*
 *  Authors:
 *    Maurizio Drocco
 *    Massimo Torquati
 *    Marco Aldinucci 
 *
 */

#ifndef FF_STENCILREDUCE_OCL_HPP
#define FF_STENCILREDUCE_OCL_HPP

#ifdef FF_OPENCL

#include <string>
#include <fstream>
#include <tuple>
#include <ff/oclnode.hpp>
#include <ff/node.hpp>
#include <ff/oclallocator.hpp>
#include <ff/stencilReduceOCL_macros.hpp>

namespace ff {

enum reduceMode { REDUCE_INPUT, REDUCE_OUTPUT };


template<typename TaskT_, typename Tin_, typename Tout_ = Tin_>
class baseOCLTask {
public:
    typedef TaskT_  TaskT;
    typedef Tin_    Tin;
    typedef Tout_   Tout;

    baseOCLTask(): inPtr(NULL),outPtr(NULL),reduceVar(NULL),
                   size_in(0),size_out(0),iter(0),
                   tuple_in(std::make_tuple(true,false,false)),
                   tuple_out(std::make_tuple(true,false,false))   { }   
    virtual ~baseOCLTask() { }
    
    // user must override this method
    virtual void setTask(const TaskT *t) = 0;
    
    /* --- the user may overrider these methods --- */ 

	virtual bool   iterCondition(const Tout&, size_t) { return false; } 
	virtual Tout   combinator(const Tout&, const Tout&)  { return Tout(); }
    virtual void   incIter()                     { ++iter; }
	virtual size_t getIter() const               { return iter; }
	virtual void   resetIter(const size_t val=0) { iter = val; } 

    /* -------------------------------------------- */ 	

    void resetTask() {
        envPtr.resize(0);
        copyEnv.resize(0);
    }
    void setInPtr(Tin* _inPtr, size_t sizeIn=1, bool copy=true, bool reuse=false, bool release=false)  { 
        inPtr  = _inPtr; size_in = sizeIn; 
        tuple_in = std::make_tuple(copy,reuse,release);
    }
    void setOutPtr(Tout* _outPtr, size_t sizeOut=0, bool copyback=true, bool reuse=false, bool release=false)  { 
        outPtr = _outPtr; size_out = sizeOut; 
        tuple_out = std::make_tuple(copyback,reuse,release);
    }
    template<typename ptrT>
    void setEnvPtr(const ptrT* _envPtr, size_t size, bool copy=true, bool reuse=false, bool release=false)  { 
        assert(envPtr.size() == copyEnv.size());
        envPtr.push_back(std::make_pair((void*)_envPtr,size*sizeof(ptrT)));
        copyEnv.push_back(std::make_tuple(sizeof(ptrT), copy,reuse,release));
    }
    
    Tin *   getInPtr()    const { return inPtr;  }
    Tout *  getOutPtr()   const { return outPtr; }
    template<typename ptrT>
    void getEnvPtr(const size_t idx, ptrT *& ptr)  const { 
        assert(idx < envPtr.size());
        ptr = reinterpret_cast<ptrT*>(envPtr[idx].first);
    }
    
    size_t  getEnvNum() const { 
        assert(envPtr.size() == copyEnv.size());
        return envPtr.size();
    }
    
    bool  getCopyEnv(const size_t idx) const { 
        assert(idx < copyEnv.size());
        return std::get<1>(copyEnv[idx]);
    }
    bool  getReuseEnv(const size_t idx) const { 
        assert(idx < copyEnv.size());
        return std::get<2>(copyEnv[idx]);
    }
    bool  getReleaseEnv(const size_t idx) const { 
        assert(idx < copyEnv.size());
        return std::get<3>(copyEnv[idx]);
    }
    
    bool getCopyIn()     const { return std::get<0>(tuple_in); }
    bool getReuseIn()    const { return std::get<1>(tuple_in); }
    bool getReleaseIn()  const { return std::get<2>(tuple_in); }

    bool getCopyOut()    const { return std::get<0>(tuple_out); }
    bool getReuseOut()   const { return std::get<1>(tuple_out); }
    bool getReleaseOut() const { return std::get<2>(tuple_out); }


    size_t getSizeIn()   const { return size_in;   }
    size_t getSizeOut()  const { return (size_out==0)?size_in:size_out;  }
    size_t getSizeEnv(const size_t idx)  const { 
        assert(idx < copyEnv.size());
        return std::get<0>(copyEnv[idx]);
    }
    
    size_t getBytesizeIn()    const { return getSizeIn() * sizeof(Tin); }
    size_t getBytesizeOut()   const { return getSizeOut() * sizeof(Tout); }
    size_t getBytesizeEnv(const size_t idx)  const { 
        assert(idx < envPtr.size());
        return envPtr[idx].second;
    }
    
    void  setReduceVar(const Tout *r) { reduceVar = (Tout*)r;  } 
    Tout *getReduceVar() const { return reduceVar;  }
	void  writeReduceVar(const Tout &r) { *reduceVar = r; }    

	void  setIdentityVal(const Tout &x) { identityVal = x;} 
	Tout  getIdentityVal() const        { return identityVal; }

	bool  iterCondition_aux() { 
        return iterCondition(*reduceVar, iter);   
    }           
protected:
    Tin     *inPtr;
    Tout    *outPtr;
    Tout    *reduceVar, identityVal;
    size_t   size_in, size_out, iter;
    std::tuple<bool,bool,bool> tuple_in;
    std::tuple<bool,bool,bool> tuple_out;

    std::vector<std::pair<void*,size_t> > envPtr;             // pointer and byte-size
    std::vector<std::tuple<size_t,bool,bool,bool> > copyEnv;  // size and flags
};


template<typename T, typename TOCL = T>
class ff_oclAccelerator {
public:
	typedef typename TOCL::Tin  Tin;
	typedef typename TOCL::Tout Tout;

	ff_oclAccelerator(ff_oclallocator &alloc, const size_t width_, const Tout &identityVal) :
        allocator(&alloc), width(width_), identityVal(identityVal), events_h2d(16), deviceId(NULL) {
		workgroup_size_map = workgroup_size_reduce = 0;
		inputBuffer = outputBuffer = reduceBuffer = NULL;

		sizeInput = sizeInput_padded = 0;
		lenInput = offset1_in = offset2_in = pad1_in = pad2_in = lenInput_global = 0;

		sizeOutput = sizeOutput_padded = 0;
		lenOutput = offset1_out = offset2_out = pad1_out = pad2_out = lenOutput_global = 0;

		nevents_h2d = nevents_map = 0;
		event_d2h = event_map = event_reduce1 = event_reduce2 = NULL;
		localThreadsMap = globalThreadsMap = 0;
		localThreadsReduce = globalThreadsReduce = numBlocksReduce = blockMemSize = 0;
		reduceVar = identityVal;
		kernel_map = kernel_reduce = kernel_init = NULL;
		context = NULL;
		program = NULL;
		cmd_queue = NULL;
	}

	void init(cl_device_id dId, const std::string &kernel_code, const std::string &kernel_name1,
              const std::string &kernel_name2) {
        deviceId = dId;
		svc_SetUpOclObjects(deviceId, kernel_code, kernel_name1,kernel_name2);
	}

	void releaseAll() { svc_releaseOclObjects(); }
    void releaseInput(const Tin *inPtr)     { 
        allocator->releaseBuffer(inPtr, context, inputBuffer);
    }
    void releaseOutput(const Tout *outPtr)  {
        allocator->releaseBuffer(outPtr, context, outputBuffer);
    }
    void releaseEnv(size_t idx, const void *envPtr) {
        allocator->releaseBuffer(envPtr, context, envBuffer[idx].first);
    }
        
	void swapBuffers() {
		cl_mem tmp = inputBuffer;
		inputBuffer = outputBuffer;
		outputBuffer = tmp;
	}

    void adjustInputBufferOffset(std::pair<size_t, size_t> &P, size_t len_global) {
        offset1_in = P.first;
        lenInput   = P.second;
        lenInput_global = len_global;
        pad1_in = (std::min)(width, offset1_in);
        offset2_in = lenInput_global - lenInput - offset1_in;
        pad2_in = (std::min)(width, offset2_in);
		sizeInput = lenInput * sizeof(Tin);
		sizeInput_padded = sizeInput + (pad1_in + pad2_in) * sizeof(Tin);
    }

	void relocateInputBuffer(const Tin *inPtr, const bool reuseIn, const Tout *reducePtr) {
		cl_int status;

        // MA patch - map not defined => workgroup_size_map
        size_t workgroup_size = workgroup_size_map==0?workgroup_size_reduce:workgroup_size_map;
                
        if (reuseIn) {
            inputBuffer = allocator->createBufferUnique(inPtr, context,
                                                       CL_MEM_READ_WRITE, sizeInput_padded, &status);
        } else {
            if (inputBuffer)  allocator->releaseBuffer(inPtr, context, inputBuffer);
            //allocate input-size + pre/post-windows
            inputBuffer = allocator->createBuffer(inPtr, context,
                                                 CL_MEM_READ_WRITE, sizeInput_padded, &status);
        }

        checkResult(status, "CreateBuffer input");
		if (lenInput < workgroup_size) {
			localThreadsMap = lenInput;
			globalThreadsMap = lenInput;
		} else {
			localThreadsMap = workgroup_size;
			globalThreadsMap = nextMultipleOfIf(lenInput, workgroup_size);
		}

        if (reducePtr) {
            // 64 and 256 are the max number of blocks and threads we want to use
            std::cerr << "Reduce inlen " << lenInput;
            getBlocksAndThreads(lenInput, 64, 256, numBlocksReduce, localThreadsReduce);
            globalThreadsReduce = numBlocksReduce * localThreadsReduce;
            //std::cerr << "numBlocksReduce " << numBlocksReduce << " localThreadsReduce " << localThreadsReduce << "\n";
            
            blockMemSize =
				(localThreadsReduce <= 32) ?
                (2 * localThreadsReduce * sizeof(Tin)) :
                (localThreadsReduce * sizeof(Tin));

            //std::cerr << "blockMemSize " << blockMemSize << "\n";
            
            if (reduceBuffer) allocator->releaseBuffer(reducePtr, context, reduceBuffer);            
            reduceBuffer = allocator->createBuffer(reducePtr, context, 
                                                   CL_MEM_READ_WRITE, numBlocksReduce * sizeof(Tin), &status);
            checkResult(status, "CreateBuffer reduce");
        }
	}

    void adjustOutputBufferOffset(std::pair<size_t, size_t> &P, size_t len_global) {
        offset1_out = P.first;
		lenOutput   = P.second;
		lenOutput_global = len_global;
		pad1_out = (std::min)(width, offset1_out);
		offset2_out = lenOutput_global - lenOutput - offset1_out;
		pad2_out = (std::min)(width, offset2_out);
		sizeOutput = lenOutput * sizeof(Tout);
		sizeOutput_padded = sizeOutput + (pad1_out + pad2_out) * sizeof(Tout);
    }

	void relocateOutputBuffer(const Tout  *outPtr) {
		cl_int status;
		if (outputBuffer) allocator->releaseBuffer(outPtr, context, outputBuffer);
		outputBuffer = allocator->createBuffer(outPtr, context,
                                               CL_MEM_READ_WRITE, sizeOutput_padded,&status);
		checkResult(status, "CreateBuffer output");
	}

	void relocateEnvBuffer(const void *envptr, const bool reuseEnv, const size_t idx, const size_t envbytesize) {
		cl_int status;

        if (idx <= envBuffer.size()) {
            cl_mem envb;
            if (reuseEnv) 
                envb = allocator->createBufferUnique(envptr, context,
                                                     CL_MEM_READ_WRITE, envbytesize, &status);
            else 
                envb = allocator->createBuffer(envptr, context,
                                               CL_MEM_READ_WRITE, envbytesize, &status);
            checkResult(status, "CreateBuffer envBuffer");
            envBuffer.push_back(std::make_pair(envb,envbytesize));
        } else { 

            if (reuseEnv) {
                envBuffer[idx].first = allocator->createBufferUnique(envptr, context,
                                                                     CL_MEM_READ_WRITE, envbytesize, &status);
            } else {
                if (envBuffer[idx].second < envbytesize) {
                    if (envBuffer[idx].first) allocator->releaseBuffer(envptr, context, envBuffer[idx].first);
                    envBuffer[idx].first = allocator->createBuffer(envptr, context,
                                                                   CL_MEM_READ_WRITE, envbytesize, &status);
                    
                }
            }
            checkResult(status, "CreateBuffer envBuffer");
            envBuffer[idx].second = envbytesize;
        }
    }
    
	void setInPlace() { 
        outputBuffer      = inputBuffer;          
        lenOutput         = lenInput;
		lenOutput_global  = lenInput_global;
		pad1_out          = pad1_in;
		pad2_out          = pad2_in;
		offset1_out       = offset1_in;
		offset2_out       = offset2_in;
		sizeOutput        = sizeInput;
		sizeOutput_padded = sizeInput_padded;
    }

	void swap() {
		cl_mem tmp = inputBuffer;
		inputBuffer = outputBuffer;
		outputBuffer = tmp;
		//set iteration-dynamic MAP kernel args
		cl_int status = clSetKernelArg(kernel_map, 0, sizeof(cl_mem), &inputBuffer);
		checkResult(status, "setKernelArg input");
		status = clSetKernelArg(kernel_map, 1, sizeof(cl_mem), &outputBuffer);
		checkResult(status, "setKernelArg output");
	}

	void setMapKernelArgs(const size_t envSize) {
		cl_uint idx = 0;
		//set iteration-dynamic MAP kernel args (init)
		cl_int status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &inputBuffer);
		checkResult(status, "setKernelArg input");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &outputBuffer);
		checkResult(status, "setKernelArg output");

		//set iteration-invariant MAP kernel args
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &lenInput_global);
		checkResult(status, "setKernelArg global input length");

		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &lenOutput);
		checkResult(status, "setKernelArg local input length");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &offset1_in);
		checkResult(status, "setKernelArg offset");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &pad1_out); // CHECK !!!
		checkResult(status, "setKernelArg pad");

        for(size_t k=0; k < envSize; ++k) {
            status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &envBuffer[k].first);
            checkResult(status, "setKernelArg env");
        }
	}

	void asyncH2Dinput(Tin *p) {
        if (nevents_h2d >= events_h2d.size()) events_h2d.reserve(nevents_h2d);
		p += offset1_in - pad1_in;
		cl_int status = clEnqueueWriteBuffer(cmd_queue, inputBuffer, CL_FALSE, 0,
                                             sizeInput_padded, p, 0, NULL, &events_h2d[nevents_h2d++]);
		checkResult(status, "copying Task to device input-buffer");
	}

	void asyncH2Denv(const size_t idx, char *p) {
        if (nevents_h2d >= events_h2d.size()) events_h2d.reserve(nevents_h2d);
		cl_int status = clEnqueueWriteBuffer(cmd_queue, envBuffer[idx].first, CL_FALSE, 0,
                                             envBuffer[idx].second, p, 0, NULL, &events_h2d[nevents_h2d++]);    
		checkResult(status, "copying Task to device env-buffer");
	}

	void asyncD2Houtput(Tout *p) {
		cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
                                            pad1_out * sizeof(Tout), sizeOutput, p + offset1_out, 0, NULL, &event_d2h);
		checkResult(status, "copying output back from device");
	}

	void asyncD2Hborders(Tout *p) {
		cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
                                            pad1_out * sizeof(Tout), width * sizeof(Tout), p + offset1_out, 0, NULL,
                                            &events_h2d[0]);
		checkResult(status, "copying border1 back from device");
		++nevents_h2d;
		status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
                                     (pad1_out + lenOutput - width) * sizeof(Tout), width * sizeof(Tout),
                                     p + offset1_out + lenOutput - width, 0, NULL, &event_d2h);
		checkResult(status, "copying border2 back from device");
	}

	void asyncH2Dborders(Tout *p) {
		if (pad1_out) {
			cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE, 0,
                                                 pad1_out * sizeof(Tout), p + offset1_out - pad1_out, 0, NULL,
                                                 &events_h2d[nevents_h2d++]);
			checkResult(status, "copying left border to device");
		}
		if (pad2_out) {
			cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE,
                                                 (pad1_out + lenOutput) * sizeof(Tout), pad2_out * sizeof(Tout),  // NOTE: in a loop Tin == Tout !!
                                                 p + offset1_out + lenOutput, 0, NULL, &events_h2d[nevents_h2d++]);
			checkResult(status, "copying right border to device");
		}
	}

	//void initReduce(const Tout &initReduceVal, reduceMode m = REDUCE_OUTPUT) {
    void initReduce(reduceMode m = REDUCE_OUTPUT) {
		//set kernel args for reduce1
		int idx = 0;
		cl_mem tmp = (m == REDUCE_OUTPUT) ? outputBuffer : inputBuffer;
		cl_int status  = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &tmp);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &pad1_in);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),	(void *) &lenInput);
		status        |= clSetKernelArg(kernel_reduce, idx++, blockMemSize, NULL);
        status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tout), (void *) &identityVal);  
		checkResult(status, "setKernelArg reduce-1");
	}

	void asyncExecMapKernel() {
		//execute MAP kernel
		cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_map, 1, NULL,
                                               &globalThreadsMap, &localThreadsMap, 0, NULL, &event_map);
		checkResult(status, "executing map kernel");
        //std::cerr << "Exec map WI " << globalThreadsMap << " localThreadsMap " << localThreadsMap << "\n";
		++nevents_map;
	}

	void asyncExecReduceKernel1() {
        //std::cerr << "Exec reduce1 WI " << globalThreadsReduce << " localThreadsReduce " << localThreadsReduce <<  "\n";
		cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
                                               &globalThreadsReduce, &localThreadsReduce, nevents_map,
                                               (nevents_map==0)?NULL:&event_map,
                                               &event_reduce1);
		checkResult(status, "exec kernel reduce-1");
		nevents_map = 0;
	}

	void asyncExecReduceKernel2() {
		cl_uint zeropad = 0;
		int idx = 0;
		cl_int status  = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &zeropad);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),(void*) &numBlocksReduce);
		status        |= clSetKernelArg(kernel_reduce, idx++, blockMemSize, NULL);
		status        |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tout), (void *) &identityVal);
		checkResult(status, "setKernelArg reduce-2");
        //std::cerr << "Exec reduce2 WI " << globalThreadsReduce << " localThreadsReduce " << localThreadsReduce <<  "\n";
		status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
                                        &localThreadsReduce, &localThreadsReduce, 1, &event_reduce1,
                                        &event_reduce2);
		checkResult(status, "exec kernel reduce-2");
	}

	Tout getReduceVar() {
		cl_int status = clEnqueueReadBuffer(cmd_queue, reduceBuffer, CL_TRUE, 0,
                                            sizeof(Tout), &reduceVar, 0, NULL, NULL);
		checkResult(status, "d2h reduceVar");
		return reduceVar;
	}

	void waitforh2d() {
        if (nevents_h2d>0) {
            cl_int status = clWaitForEvents(nevents_h2d, events_h2d.data());
            checkResult(status, "h2d wait for");
            nevents_h2d = 0;
        }
	}

	void waitford2h() {
		cl_int status = clWaitForEvents(1, &event_d2h);
		checkResult(status, "d2h wait for");
	}

	void waitforreduce() {
		cl_int status = clWaitForEvents(1, &event_reduce2);
        checkResult(status, "wait for reduce");
	}

	void waitformap() {
		cl_int status = clWaitForEvents(nevents_map, &event_map);
		nevents_map = 0;
        checkResult(status, "wait for map");
	}

private:
	void buildKernelCode(const std::string &kc, cl_device_id dId) {
		cl_int status;

		size_t sourceSize = kc.length();
		const char* code = kc.c_str();

		//printf("code=\n%s\n", code);
		program = clCreateProgramWithSource(context, 1, &code, &sourceSize, &status);
		checkResult(status, "creating program with source");

		status = clBuildProgram(program, 1, &dId, /*"-cl-fast-relaxed-math"*/NULL, NULL,NULL);
		checkResult(status, "building program");

		//#if 0
		// DEBUGGING CODE for checking OCL compilation errors
		if (status != CL_SUCCESS) {
			printf("\nFail to build the program\n");
			size_t len;
			clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
			printf("LOG len %ld\n", len);
			char *buffer = (char*) calloc(len, sizeof(char));
			assert(buffer);
			clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, len * sizeof(char),
					buffer, NULL);
			printf("LOG: %s\n\n", buffer);
		}
		//#endif
	}

	void svc_SetUpOclObjects(cl_device_id dId, const std::string &kernel_code,
                             const std::string &kernel_name1, const std::string &kernel_name2) {

		cl_int status;
        const oclParameter *param = clEnvironment::instance()->getParameter(dId);
        assert(param);
        context    = param->context;
        cmd_queue  = param->commandQueue;

		//compile kernel on device
		buildKernelCode(kernel_code, dId);
        
        char buf[128];
        size_t s, ss[3];
        
        clGetDeviceInfo(dId, CL_DEVICE_NAME, 128, buf, NULL);
        clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &s, NULL);
        clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), &ss, NULL);
        std::cerr << "svc_SetUpOclObjects dev " << dId << " " << buf << " MAX_WORK_GROUP_SIZE " << s << " MAX_WORK_ITEM_SIZES " << ss[0] << " " << ss[1] << " " << ss[2] << "\n";
        
        
		//create kernel objects
		if (kernel_name1 != "") {
			kernel_map = clCreateKernel(program, kernel_name1.c_str(), &status);
			checkResult(status, "CreateKernel (map)");
			status = clGetKernelWorkGroupInfo(kernel_map, dId,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_map, 0);
			checkResult(status, "GetKernelWorkGroupInfo (map)");
            //std::cerr << "GetKernelWorkGroupInfo (map) " << workgroup_size_map << "\n";
		}

		if (kernel_name2 != "") {
			kernel_reduce = clCreateKernel(program, kernel_name2.c_str(), &status);
			checkResult(status, "CreateKernel (reduce)");
			status = clGetKernelWorkGroupInfo(kernel_reduce, dId,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_reduce, 0);
			checkResult(status, "GetKernelWorkGroupInfo (reduce)");
            //std::cerr << "GetKernelWorkGroupInfo (reduce) " << workgroup_size_reduce << "\n";
		}

	}

	void svc_releaseOclObjects() {
		if (kernel_map)     clReleaseKernel(kernel_map);
		if (kernel_reduce)	clReleaseKernel(kernel_reduce);
		clReleaseProgram(program);

		// if (inputBuffer)    clReleaseMemObject(inputBuffer);
        // for(size_t i=0; i < envBuffer.size(); ++i)
        //     clReleaseMemObject(envBuffer[i].first);
		// if (outputBuffer && outputBuffer != inputBuffer)
		// 	clReleaseMemObject(outputBuffer);
		// if (reduceBuffer)
		// 	clReleaseMemObject(reduceBuffer);
        allocator->releaseAllBuffers(context);
	}

	/*!
	 * Computes the number of threads and blocks to use for the reduction kernel.
	 */
	inline void getBlocksAndThreads(const size_t size, const size_t maxBlocks,
			const size_t maxThreads, size_t & blocks, size_t &threads) {
		const size_t half = (size + 1) / 2;
		threads =
				(size < maxThreads * 2) ?
						(isPowerOf2(half) ? nextPowerOf2(half + 1) : nextPowerOf2(half)) :
						maxThreads;
		blocks = (size + (threads * 2 - 1)) / (threads * 2);
		blocks = std::min(maxBlocks, blocks);
	}

protected:  
	cl_context context;
	cl_program program;
	cl_command_queue cmd_queue;
	cl_mem reduceBuffer;
	cl_kernel kernel_map, kernel_reduce, kernel_init;
private:
    ff_oclallocator *allocator;
	const size_t width;
	const Tout identityVal;
	Tout reduceVar;

	cl_mem inputBuffer, outputBuffer;
    std::vector<std::pair<cl_mem, size_t> > envBuffer;

	size_t sizeInput, sizeInput_padded;
	size_t lenInput, offset1_in, offset2_in, pad1_in, pad2_in, lenInput_global;

	size_t sizeOutput, sizeOutput_padded;
	size_t lenOutput, offset1_out, offset2_out, pad1_out, pad2_out, lenOutput_global;


	size_t workgroup_size_map, workgroup_size_reduce;
	size_t nevents_h2d, nevents_map;
	size_t localThreadsMap, globalThreadsMap;
	size_t localThreadsReduce, globalThreadsReduce, numBlocksReduce, blockMemSize;
	cl_event event_d2h, event_map, event_reduce1, event_reduce2;

    std::vector<cl_event> events_h2d;

    cl_device_id deviceId; 
};


/*!
 * \class ff_stencilReduceLoopOCL_1D
 * \ingroup high_level_patterns
 *
 * \brief The OpenCL-based StencilReduceLoop pattern in 1 dimension
 *
 * This class is defined in \ref stencilReduceOCL.hpp
 */
    
template<typename T, typename TOCL = T>
class ff_stencilReduceLoopOCL_1D: public ff_oclNode_t<T> {
public:
	typedef typename TOCL::Tin         Tin;
	typedef typename TOCL::Tout        Tout;
	typedef ff_oclAccelerator<T, TOCL> accelerator_t;

	ff_stencilReduceLoopOCL_1D(const std::string &mapf,			              //OCL elemental
                               const std::string &reducef = std::string(""),  //OCL combinator
                               const Tout &identityVal = Tout(),
                               const ff_oclallocator &allocator = ff_oclallocator(),
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(false), accelerators(NACCELERATORS, accelerator_t(const_cast<ff_oclallocator&>(allocator), width,identityVal)), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
    stencil_width(width), preferred_dev(0), oldSizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
		setcode(mapf, reducef);
	}

	//template <typename Function>
	ff_stencilReduceLoopOCL_1D(const T &task, const std::string &mapf,	                //OCL elemental
                               const std::string &reducef = std::string(""),			//OCL combinator
                               const Tout &identityVal = Tout(),
                               const ff_oclallocator &allocator = ff_oclallocator(),
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(true), accelerators(NACCELERATORS, accelerator_t(const_cast<ff_oclallocator&>(allocator), width,identityVal)), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
        stencil_width(width), oldSizeIn(0), oldSizeOut(0), oldSizeReduce(0) {
		ff_node::skipfirstpop(true);
		setcode(mapf, reducef);
		Task.setTask(const_cast<T*>(&task));
	}

	virtual ~ff_stencilReduceLoopOCL_1D() {}

    // used to set tasks when in onshot mode 
    void setTask(const T &task) { 
        assert(oneshot);
        Task.resetTask();
        Task.setTask(&task); 
    }

    // sets the OpenCL devices that have to be used
    int setDevices(std::vector<cl_device_id> &dev) {
        if (dev.size() > accelerators.size()) return -1; 
        devices = dev;
        if (ff_oclNode_t<T>::oclId < 0) 
            ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();

        for (size_t i = 0; i < devices.size(); ++i) 
            accelerators[i].init(devices[i], kernel_code, kernel_name1,kernel_name2);

        return 0;
    }

    // force execution on the CPU
    void pickCPU () {
        std::cerr << "select CPU\n";
        ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_CPU);
        
    }
    
    // force execution on the GPU
    void pickGPU (size_t offset=0 /* picks first GPU */) {
        std::cerr << "select GPU\n";
        preferred_dev=offset;
        ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_GPU);
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

	const T* getTask() const {
		return &Task;
	}

	unsigned int getIter() {
		return Task.getIter();
	}

	Tout *getReduceVar() {
		assert(oneshot);
		return Task.getReduceVar();
	}
/*
    int nodeInit() { 
        if (devices.size()==0) {
            ssize_t devId;
            switch(ff_oclNode_t<T>::getDeviceType()) {
            case CL_DEVICE_TYPE_CPU: {                
                devId = clEnvironment::instance()->getCPUDevice();
                if (accelerators.size()!=1)  printf("WARNING: Too many accelerators rquested for CL_CPU device\n");
                ff_oclNode_t<T>::deviceId=clEnvironment::instance()->getDevice(devId);
            } break;
            case CL_DEVICE_TYPE_GPU: {
                devId = clEnvironment::instance()->getGPUDevice();
                if (accelerators.size()!=1)  printf("WARNING: Too many accelerators rquested for CL_GPU device\n");
                ff_oclNode_t<T>::deviceId=clEnvironment::instance()->getDevice(devId);
            } break;
            default: {
                error("stencilReduceOCL::nodeInit: no device selected\n");
                return -1;
            }
            }
            accelerators[0].init(ff_oclNode_t<T>::deviceId, kernel_code, kernel_name1,kernel_name2); 
            return 0;
        }
        if (devices.size() > accelerators.size()) return -1; 
        for (size_t i = 0; i < devices.size(); ++i) 
            accelerators[i].init(devices[i], kernel_code, kernel_name1,kernel_name2);
        return 0;
    }
*/
    
    int nodeInit() {
        bool device_found = false;
        if (devices.size()==0) {
            switch(ff_oclNode_t<T>::getDeviceType()) {
                case CL_DEVICE_TYPE_ALL:
                    fprintf(stderr,"STATUS: requested ALL\n");
                case CL_DEVICE_TYPE_GPU: {// One or more GPUs
                    std::vector<ssize_t> devIds = clEnvironment::instance()->getAllGPUDevices();
                    if (devIds.size()==0) {
                        fprintf(stderr,"WARNING: no GPUs available\n");
                    } else {
                        if (accelerators.size()==1) {
                            ssize_t devId = devIds[preferred_dev];
                            ff_oclNode_t<T>::deviceId=clEnvironment::instance()->getDevice(devId);
                            accelerators[0].init(ff_oclNode_t<T>::deviceId, kernel_code, kernel_name1,kernel_name2);
                            device_found = true;
                        } else {
                            if (devIds.size()>=accelerators.size()) {
                                for (size_t j=0; accelerators.size(); ++j)
                                    accelerators[j].init(clEnvironment::instance()->getDevice(devIds[j]), kernel_code, kernel_name1,kernel_name2);
                                device_found = true;
                            } else {
                                fprintf(stderr,"WARNING: requested number of devices not matching number of logical accelerators\n");
                            }
                        }
                    }
                } if (device_found) break;
                case CL_DEVICE_TYPE_CPU: {
                    fprintf(stderr,"STATUS: requested 1CPU\n");
                    ssize_t devId = clEnvironment::instance()->getCPUDevice();
                    if (accelerators.size()!=1)  fprintf(stderr,"WARNING: Too many accelerators requested for CL_CPU device\n");
                    // accelerators num should be modified
                    ff_oclNode_t<T>::deviceId=clEnvironment::instance()->getDevice(devId);
                    accelerators[0].init(ff_oclNode_t<T>::deviceId, kernel_code, kernel_name1,kernel_name2);
                    device_found = true;
                } break;
                default: {
                    error("stencilReduceOCL::nodeInit: no device selected\n");
                    return -1;
                }
                    
            }
        }
        return (device_found?0:-1);
    }
    
        
    void nodeEnd() {
        if (devices.size()==0) {
            if (ff_oclNode_t<T>::deviceId != NULL)
                accelerators[0].releaseAll();
        } else 
            for (size_t i = 0; i < devices.size(); ++i) 
                accelerators[i].releaseAll();     
    }
     
protected:

	virtual bool isPureMap() const    { return false; }
	virtual bool isPureReduce() const { return false; }

    /* static mapping
     * Performs a static GPU greedy algorithm to allocate openCL resources
     */
    /*
    virtual int svc_init() {
        if (ff_oclNode_t<T>::oclId < 0) {
            ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();
        nodeInit();
        }
        return 0;
    }
    */
    
    // To be rewritten in terms of nodeinit 
	virtual int svc_init() {
        if (ff_oclNode_t<T>::oclId < 0) {
            ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();
            
            if (devices.size()>0) {
                if (devices.size() > accelerators.size()) {
                    error("stencilReduceOCL::svc_init: more devices than accelerators !\n");
                    return -1; 
                }
                for (size_t i = 0; i < devices.size(); ++i) 
                    accelerators[i].init(devices[i], kernel_code, kernel_name1,kernel_name2);                
            } else {
                cl_device_id deviceId; 
                
                // Forced devices
                if (ff_oclNode_t<T>::getDeviceType()==CL_DEVICE_TYPE_CPU) {
                    ssize_t CPUdevId = clEnvironment::instance()->getCPUDevice();
                    //fprintf(stderr,"%d: Allocated a CPU device as either no GPU device is available or no GPU slot is available (cpuId=%ld)\n",ff_oclNode_t<T>::oclId, CPUdevId);
                    if (accelerators.size()!=1)  printf("WARNING: Too many accelerators rquested for CL_CPU device\n");
                    deviceId=clEnvironment::instance()->getDevice(CPUdevId);
                    accelerators[0].init(deviceId, kernel_code, kernel_name1,kernel_name2);
                } else {
                    // TO BE REVIEWED
                    
                    
                    for (size_t i = 0; i < accelerators.size(); ++i) {
                        ssize_t GPUdevId =clEnvironment::instance()->getGPUDeviceRR();
                        if( (GPUdevId !=-1) && ( ff_oclNode_t<T>::oclId < clEnvironment::instance()->getNumGPU())) { 
                            //fprintf(stderr,"%d: Allocated a GPU device for accelerator %ld, the id is %ld\n", ff_oclNode_t<T>::oclId, i, GPUdevId);
                            deviceId=clEnvironment::instance()->getDevice(GPUdevId);
                        } else  {	
                            // fall back to CPU either GPU has reached its max or there is no GPU available
                            ssize_t CPUdevId =clEnvironment::instance()->getCPUDevice();
                            //fprintf(stderr,"%d: Allocated a CPU device as either no GPU device is available or no GPU slot is available (cpuId=%ld)\n",ff_oclNode_t<T>::oclId, CPUdevId);
                            deviceId=clEnvironment::instance()->getDevice(CPUdevId);
                        }
                        accelerators[i].init(deviceId, kernel_code, kernel_name1,kernel_name2);
                    }
                }
            }
        }
        return 0;
    }
    

	virtual void svc_end() {
		if (!ff::ff_node::isfrozen()) nodeEnd();
	}

	T *svc(T *task) {
		if (task) {
            Task.resetTask();
            Task.setTask(task);
        }
		Tin   *inPtr = Task.getInPtr();
		Tout  *outPtr = Task.getOutPtr();
        Tout  *reducePtr = Task.getReduceVar();
        const size_t envSize = Task.getEnvNum();
        
		// relocate input device memory if needed
		if (oldSizeIn != Task.getBytesizeIn()) {
			compute_accmem(Task.getSizeIn(),  acc_in);

            for (size_t i = 0; i < accelerators.size(); ++i) {
                accelerators[i].adjustInputBufferOffset(acc_in[i], Task.getSizeIn());
            }

            if (oldSizeIn < Task.getBytesizeIn()) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i].relocateInputBuffer(inPtr, Task.getReuseIn(), reducePtr);
                }
                oldSizeIn = Task.getBytesizeIn();
            }            
		}

		// relocate output device memory if needed
        if (oldSizeOut != Task.getBytesizeOut()) {

            if ((void*)inPtr == (void*)outPtr) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i].setInPlace();
                }
            } else {
                compute_accmem(Task.getSizeOut(), acc_out);

                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i].adjustOutputBufferOffset(acc_out[i], Task.getSizeOut());
                }
                
                if (oldSizeOut < Task.getBytesizeOut()) {
                    for (size_t i = 0; i < accelerators.size(); ++i) {
                        accelerators[i].relocateOutputBuffer(outPtr); 
                    }
                    oldSizeOut = Task.getBytesizeOut();
                }
            }
        }


        /* NOTE: env buffer are replicated on all devices.
         *       It would be nice to have different polices: replicated, partitioned, etc....
         *
         */  
        for (size_t i = 0; i < accelerators.size(); ++i)
            for(size_t k=0; k < envSize; ++k) {
                char *envptr;
                Task.getEnvPtr(k, envptr);
                accelerators[i].relocateEnvBuffer(envptr, Task.getReuseEnv(k), k, Task.getBytesizeEnv(k));
            }        

		if (!isPureReduce())  //set kernel args
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i].setMapKernelArgs(envSize);

		//(async) copy input and environments (h2d)
		for (size_t i = 0; i < accelerators.size(); ++i) {

            if (Task.getCopyIn())
                accelerators[i].asyncH2Dinput(Task.getInPtr());  //in

            for(size_t k=0; k < envSize; ++k) {
                if (Task.getCopyEnv(k)) {
                    char *envptr;
                    Task.getEnvPtr(k, envptr);
                    accelerators[i].asyncH2Denv(k, envptr);
                }
            }
        } 

		if (isPureReduce()) {
			//init reduce
			for (size_t i = 0; i < accelerators.size(); ++i)
                accelerators[i].initReduce(REDUCE_INPUT);

			//wait for cross-accelerator h2d
			waitforh2d();

			//(async) device-reduce1
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i].asyncExecReduceKernel1();

			//(async) device-reduce2
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i].asyncExecReduceKernel2();

			waitforreduce(); //wait for cross-accelerator reduce

			//host-reduce
			Tout redVar = accelerators[0].getReduceVar();
			for (size_t i = 1; i < accelerators.size(); ++i)
				redVar = Task.combinator(redVar, accelerators[i].getReduceVar());
			Task.writeReduceVar(redVar);
		} else {
			Task.resetIter();

			if (isPureMap()) {
				//wait for cross-accelerator h2d
				waitforh2d();

				//(async) exec kernel
				for (size_t i = 0; i < accelerators.size(); ++i)
					accelerators[i].asyncExecMapKernel();
				Task.incIter();

				waitformap(); //join

			} else { //iterative Map-Reduce (aka stencilReduceLoop)

                //invalidate first swap
				for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i].swap();

				bool go = true;
				do {
					//Task.before();

					for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i].swap();

					//wait for cross-accelerator h2d
					waitforh2d();

					//(async) execute MAP kernel
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i].asyncExecMapKernel();
					Task.incIter();

					//start async-interleaved: reduce + borders sync
					//init reduce
					for (size_t i = 0; i < accelerators.size(); ++i)
                        accelerators[i].initReduce();

					//(async) device-reduce1
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i].asyncExecReduceKernel1();

					//(async) device-reduce2
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i].asyncExecReduceKernel2();

					//wait for cross-accelerators reduce
					waitforreduce();

					//host-reduce
					Tout redVar = accelerators[0].getReduceVar();
					for (size_t i = 1; i < accelerators.size(); ++i) 
						redVar = Task.combinator(redVar, accelerators[i].getReduceVar());
                    Task.writeReduceVar(redVar);

					go = Task.iterCondition_aux();
					if (go) {
                        assert(outPtr);
						//(async) read back borders (d2h)
						for (size_t i = 0; i < accelerators.size(); ++i)
							accelerators[i].asyncD2Hborders(outPtr);
						waitford2h(); //wait for cross-accelerators d2h
						//(async) read borders (h2d)
						for (size_t i = 0; i < accelerators.size(); ++i)
							accelerators[i].asyncH2Dborders(outPtr);
					}

					//Task.after();

				} while (go);
			}
            
            //(async)read back output (d2h)
            if (outPtr && Task.getCopyOut()) { // do we have to copy back the output result ?
                for (size_t i = 0; i < accelerators.size(); ++i)
                    accelerators[i].asyncD2Houtput(outPtr);
                waitford2h(); //wait for cross-accelerators d2h
            }
		}

        if (Task.getReleaseIn()) {
            for (size_t i = 0; i < accelerators.size(); ++i) 
                accelerators[i].releaseInput(inPtr);
        }
        if (Task.getReleaseOut()) {
            for (size_t i = 0; i < accelerators.size(); ++i) 
                accelerators[i].releaseOutput(outPtr);
        }

        for(size_t k=0; k < envSize; ++k) {
            if (Task.getReleaseEnv(k)) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    char *envptr;
                    Task.getEnvPtr(k, envptr);
                    accelerators[i].releaseEnv(k,outPtr);
                }
            }
        }
        
		return (oneshot ? NULL : task);
	}

private:
	void setcode(const std::string &codestr1, const std::string &codestr2) {
		int n = 0;
		if (codestr1 != "") {
			n = codestr1.find_first_of("|");
			assert(n > 0);
			kernel_name1 = codestr1.substr(0, n);
			const std::string &tmpstr = codestr1.substr(n + 1);
			n = tmpstr.find_first_of("|");
			assert(n > 0);

			// checking for double type
			if (tmpstr.substr(0, n) == "double") {
				kernel_code = "\n#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n"
						+ tmpstr.substr(n + 1);
			} else
				kernel_code = "\n" + tmpstr.substr(n + 1);
		}

		// checking for extra code needed to compile the kernels
		std::ifstream ifs(FF_OPENCL_DATATYPES_FILE);
		if (ifs.is_open())
			kernel_code.insert(kernel_code.begin(), std::istreambuf_iterator<char>(ifs),
					std::istreambuf_iterator<char>());

		if (codestr2 != "") {
			n = codestr2.find("|");
			assert(n > 0);
			kernel_name2 += codestr2.substr(0, n);
			const std::string &tmpstr = codestr2.substr(n + 1);
			n = tmpstr.find("|");
			assert(n > 0);

			// checking for double type
			if (tmpstr.substr(0, n) == "double") {
				kernel_code += "#pragma OPENCL EXTENSION cl_khr_fp64: enable\n"
						+ tmpstr.substr(n + 1);
			} else
				kernel_code += tmpstr.substr(n + 1);
		}
	}

	void compute_accmem(const size_t len, std::vector<std::pair<size_t,size_t> > &acc) {
		size_t start = 0, step = (len + accelerators.size() - 1) / accelerators.size();
		int i = 0;
		for (; i < accelerators.size() - 1; ++i) {
            acc[i]=std::make_pair(start, step);
			start += step;
		}
        acc[i]=std::make_pair(start, len-start);
	}

	void waitforh2d() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i].waitforh2d();
	}

	void waitford2h() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i].waitford2h();
	}

	void waitforreduce() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i].waitforreduce();
	}

	void waitformap() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i].waitformap();
	}

private:
	TOCL Task;
	const bool oneshot;

    std::vector<accelerator_t> accelerators;
    std::vector<std::pair<size_t, size_t> > acc_in;
    std::vector<std::pair<size_t, size_t> > acc_out;
    std::vector<cl_device_id> devices;
	int stencil_width;
    size_t preferred_dev;

	std::string kernel_code;
	std::string kernel_name1;
	std::string kernel_name2;

    size_t forced_cpu;
    size_t forced_gpu;
    size_t forced_other;
    
    
	size_t oldSizeIn, oldSizeOut, oldSizeReduce;
};


/*!
 * \class ff_mapOCL_1D
 *  \ingroup high_level_patterns
 *
 * \brief The OpenCL-based Map pattern in 1 dimension
 *
 * This class is defined in \ref stencilReduceOCL.hpp
 *
 */
template<typename T, typename TOCL = T>
class ff_mapOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
	ff_mapOCL_1D(std::string mapf, const ff_oclallocator &alloc= ff_oclallocator(), 
                 const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, "", 0, alloc, NACCELERATORS, 0) {
	}

	ff_mapOCL_1D(const T &task, std::string mapf, 
                 const ff_oclallocator &alloc= ff_oclallocator(), 
                 const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, "", 0, alloc, NACCELERATORS, 0) {
	}
	bool isPureMap() const { return true; }
};

/*!
 * \class ff_reduceOCL_1D
 *
 * \ingroup high_level_patterns
 *
 * \brief The OpenCL-based Reduce pattern in 1 dimension
 *
 * This class is defined in \ref stencilReduceOCL.hpp
 *
 */
template<typename T, typename TOCL = T>
class ff_reduceOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
	typedef typename TOCL::Tin  Tin;
	typedef typename TOCL::Tout Tout;

	ff_reduceOCL_1D(std::string reducef, const Tout &identityVal = Tout(),
                    const ff_oclallocator &alloc= ff_oclallocator(), 
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>("", reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
	ff_reduceOCL_1D(const T &task, std::string reducef, const Tout identityVal = Tout(),
                    const ff_oclallocator &alloc= ff_oclallocator(), 
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, "", reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
	bool isPureReduce() const { return true; }
};

/*
 * \class f_mapReduceOCL_1D
 *  \ingroup high_level_patterns
 *
 * \brief The mapReduce skeleton.
 *
 * The mapReuce skeleton using OpenCL
 *
 * This class is defined in \ref map.hpp
 *
 */
template<typename T, typename TOCL = T>
class ff_mapReduceOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
	typedef typename TOCL::Tin  Tin;
	typedef typename TOCL::Tout Tout;

	ff_mapReduceOCL_1D(std::string mapf, std::string reducef, const Tout &identityVal = Tout(),
                       const ff_oclallocator &alloc= ff_oclallocator(), 
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
	}

	ff_mapReduceOCL_1D(const T &task, std::string mapf, std::string reducef,
                       const Tout &identityVal = Tout(), 
                       const ff_oclallocator &alloc= ff_oclallocator(), 
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
};

}
// namespace ff

#endif // FF_OCL
#endif /* FF_MAP_OCL_HPP */

