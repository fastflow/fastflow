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
    
    // user must override this method using:
    // - setInPtr for setting the host-pointer to the input array
    // - setOutPtr for setting the host-pointer to the output array
    // - setEnvPtr for adding to env-list the host-pointer to a read-only env
    // NOTE: order of setEnvPtr calls matters! TODO refine interface?
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

    /**
     * set the host-pointer to the input array.
     *
     * @param _inPtr the host-pointer
     * @param sizeIn the number of elements in the input array
     * @param copy TODO
     * @param reuse TODO
     * @param release TODO
     */
    void setInPtr(Tin* _inPtr, size_t sizeIn=1, bool copy=true, bool reuse=false, bool release=false)  { 
        inPtr  = _inPtr; size_in = sizeIn; 
        tuple_in = std::make_tuple(copy,reuse,release);
    }

    /**
     * set the host-pointer to the output array.
     *
     * @see setInPtr()
     * @param copyback TODO
     */
    void setOutPtr(Tout* _outPtr, size_t sizeOut=0, bool copyback=true, bool reuse=false, bool release=false)  { 
        outPtr = _outPtr; size_out = sizeOut; 
        tuple_out = std::make_tuple(copyback,reuse,release);
    }

    /**
     * add to env-list the host-pointer to a read-only env.
     *
     * @see setInPtr()
     */
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

	ff_oclAccelerator(ff_oclallocator *alloc, const size_t width_, const Tout &identityVal, const bool from_source=false) :
        from_source(from_source), my_own_allocator(false), allocator(alloc), width(width_), identityVal(identityVal), events_h2d(16), deviceId(NULL) {
		workgroup_size_map = workgroup_size_reduce = 0;
		inputBuffer = outputBuffer = reduceBuffer = NULL;

		sizeInput = sizeInput_padded = 0;
		lenInput = offset1_in = pad1_in = pad2_in = lenInput_global = 0;

		sizeOutput = sizeOutput_padded = 0;
		lenOutput = offset1_out = pad1_out = pad2_out = lenOutput_global = 0;

		nevents_h2d = nevents_map = 0;
		event_d2h = event_map = event_reduce1 = event_reduce2 = NULL;
		localThreadsMap = globalThreadsMap = 0;
		localThreadsReduce = globalThreadsReduce = numBlocksReduce = blockMemSize = 0;
		reduceVar = identityVal;
		kernel_map = kernel_reduce = kernel_init = NULL;
		context = NULL;
		program = NULL;
		cmd_queue = NULL;

        if (!allocator) {
            my_own_allocator = true;
            allocator = new ff_oclallocator;
            assert(allocator);
        }
	}

    ~ff_oclAccelerator() {
        if (my_own_allocator) {
            delete allocator;
            allocator = NULL;
            my_own_allocator = false;
        }
    }

	int init(cl_device_id dId, const std::string &kernel_code, const std::string &kernel_name1,
             const std::string &kernel_name2, const bool save_binary, const bool reuse_binary) {
        deviceId = dId;
		return svc_SetUpOclObjects(deviceId, kernel_code, kernel_name1,kernel_name2, 
                                   from_source, save_binary, reuse_binary);
	}

	void releaseAll() { if (deviceId) { svc_releaseOclObjects(); deviceId = NULL; }}
    void releaseInput(const Tin *inPtr)     { 
        if (allocator->releaseBuffer(inPtr, context, inputBuffer) != CL_SUCCESS)
            checkResult(CL_INVALID_MEM_OBJECT, "releaseInput");
        inputBuffer = NULL;
    }
    void releaseOutput(const Tout *outPtr)  {
        if (allocator->releaseBuffer(outPtr, context, outputBuffer) != CL_SUCCESS)
            checkResult(CL_INVALID_MEM_OBJECT, "releaseOutput");
        outputBuffer = NULL;
    }
    void releaseEnv(size_t idx, const void *envPtr) {
        if (allocator->releaseBuffer(envPtr, context, envBuffer[idx].first) != CL_SUCCESS)
            checkResult(CL_INVALID_MEM_OBJECT, "releaseEnv");
        envBuffer[idx].first = NULL, envBuffer[idx].second = 0;
    }
        
	void swapBuffers() {
		cl_mem tmp = inputBuffer;
		inputBuffer = outputBuffer;
		outputBuffer = tmp;
	}

	//see comments for members
    void adjustInputBufferOffset(const Tin *newPtr, const Tin *oldPtr, std::pair<size_t, size_t> &P, size_t len_global) {
        offset1_in = P.first;
        lenInput   = P.second;
        lenInput_global = len_global;
        pad1_in = (std::min)(width, offset1_in);
        pad2_in = (std::min)(width, lenInput_global - lenInput - offset1_in);
		sizeInput = lenInput * sizeof(Tin);
		sizeInput_padded = sizeInput + (pad1_in + pad2_in) * sizeof(Tin);

        if (oldPtr != NULL) allocator->updateKey(oldPtr, newPtr, context);
    }

	void relocateInputBuffer(const Tin *inPtr, const bool reuseIn, const Tout *reducePtr) {
		cl_int status;

        // MA patch - map not defined => workgroup_size_map
        size_t workgroup_size = workgroup_size_map==0?workgroup_size_reduce:workgroup_size_map;
                
        if (reuseIn) {
            inputBuffer = allocator->createBufferUnique(inPtr, context,
                                                       CL_MEM_READ_WRITE, sizeInput_padded, &status);
            checkResult(status, "CreateBuffer(Unique) input");
        } else {
            if (inputBuffer)  allocator->releaseBuffer(inPtr, context, inputBuffer);
            //allocate input-size + pre/post-windows
            inputBuffer = allocator->createBuffer(inPtr, context,
                                                 CL_MEM_READ_WRITE, sizeInput_padded, &status);
            checkResult(status, "CreateBuffer input");
        }


		if (lenInput < workgroup_size) {
			localThreadsMap = lenInput;
			globalThreadsMap = lenInput;
		} else {
			localThreadsMap = workgroup_size;
			globalThreadsMap = nextMultipleOfIf(lenInput, workgroup_size);
		}

        if (reducePtr) {
            // 64 and 256 are the max number of blocks and threads we want to use
            //std::cerr << "Reduce inlen " << lenInput;
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

    void adjustOutputBufferOffset(const Tout *newPtr, const Tout *oldPtr, std::pair<size_t, size_t> &P, size_t len_global) {
        offset1_out = P.first;
		lenOutput   = P.second;
		lenOutput_global = len_global;
		pad1_out = (std::min)(width, offset1_out);
		pad2_out = (std::min)(width, lenOutput_global - lenOutput - offset1_out);
		sizeOutput = lenOutput * sizeof(Tout);
		sizeOutput_padded = sizeOutput + (pad1_out + pad2_out) * sizeof(Tout);

        if (oldPtr != NULL) allocator->updateKey(oldPtr, newPtr, context);
    }

	void relocateOutputBuffer(const Tout  *outPtr) {
		cl_int status;
		if (outputBuffer) allocator->releaseBuffer(outPtr, context, outputBuffer);
		outputBuffer = allocator->createBuffer(outPtr, context,
                                               CL_MEM_READ_WRITE, sizeOutput_padded,&status);
		checkResult(status, "CreateBuffer output");
	}

	void relocateEnvBuffer(const void *envptr, const bool reuseEnv, const size_t idx, const size_t envbytesize) {
		cl_int status = CL_SUCCESS;

        if (idx >= envBuffer.size()) {
            cl_mem envb;
            if (reuseEnv) 
                envb = allocator->createBufferUnique(envptr, context,
                                                     CL_MEM_READ_WRITE, envbytesize, &status);
            else 
                envb = allocator->createBuffer(envptr, context,
                                               CL_MEM_READ_WRITE, envbytesize, &status);
            if (checkResult(status, "CreateBuffer envBuffer"))
                envBuffer.push_back(std::make_pair(envb,envbytesize));
        } else { 

            if (reuseEnv) {
                envBuffer[idx].first = allocator->createBufferUnique(envptr, context,
                                                                     CL_MEM_READ_WRITE, envbytesize, &status);
                if (checkResult(status, "CreateBuffer envBuffer"))
                    envBuffer[idx].second = envbytesize;
            } else {
                if (envBuffer[idx].second < envbytesize) {
                    if (envBuffer[idx].first) allocator->releaseBuffer(envptr, context, envBuffer[idx].first);
                    envBuffer[idx].first = allocator->createBuffer(envptr, context,
                                                                   CL_MEM_READ_WRITE, envbytesize, &status);
                    if (checkResult(status, "CreateBuffer envBuffer"))
                        envBuffer[idx].second = envbytesize;
                }
            }
        }
    }
    
	void setInPlace() { 
        outputBuffer      = inputBuffer;          
        lenOutput         = lenInput;
		lenOutput_global  = lenInput_global;
		pad1_out          = pad1_in;
		pad2_out          = pad2_in;
		offset1_out       = offset1_in;
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
    int buildProgram(cl_device_id dId) {
		cl_int status = clBuildProgram(program, 1, &dId, /*"-cl-fast-relaxed-math"*/NULL, NULL,NULL);
		checkResult(status, "building program");

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
            
            return -1;
		}
        return 0;
    }

	int buildKernelCode(const std::string &kc, cl_device_id dId) {
		cl_int status;

		size_t sourceSize = kc.length();
		const char* code = kc.c_str();

		//printf("code=\n%s\n", code);
		program = clCreateProgramWithSource(context, 1, &code, &sourceSize, &status);
        if (!program) {
            checkResult(status, "creating program with source");
            return -1;
        }
        return buildProgram(dId);
	}

    // create the program with the binary file or from the source code
	int createProgram(const std::string &filepath, cl_device_id dId, const bool save_binary, const bool reuse_binary) {
        cl_int status, binaryStatus;
        const std::string binpath = filepath + ".bin";

        std::ifstream ifs;
        if (reuse_binary) {
            ifs.open(binpath, std::ios::binary );
            if (!ifs.is_open()) { // try with filepath
                ifs.open(filepath, std::ios::binary);
                if (!ifs.is_open()) {
                    error("createProgram: cannot open %s (nor %s)\n", filepath.c_str(), binpath.c_str());
                    return -1;
                }
            }
        } else {
            ifs.open(filepath, std::ios::binary);
            if (!ifs.is_open()) {
                error("createProgram: cannot open source file %s\n", filepath.c_str());
                return -1;
            }
        }
        std::vector<char> buf((std::istreambuf_iterator<char>(ifs)),
                              (std::istreambuf_iterator<char>()));
        ifs.close();
        size_t bufsize      = buf.size();
        const char *bufptr  = buf.data();
        program = clCreateProgramWithBinary(context, 1, &dId, &bufsize,
                                            reinterpret_cast<const unsigned char **>(&bufptr),
                                            &binaryStatus, &status);
        
        if (status != CL_SUCCESS) { // maybe is not a binary file
            program = clCreateProgramWithSource(context, 1,&bufptr, &bufsize, &status);
            if (!program) {
                checkResult(status, "creating program with source");
                return -1;  
            }
            if (buildProgram(dId)<0) return -1;
            if (save_binary) { // TODO: the logical deviceId has to be attached to the file name !                
                size_t programBinarySize;
                status = clGetProgramInfo(program, CL_PROGRAM_BINARY_SIZES,
                                          sizeof(size_t) * 1,
                                          &programBinarySize, NULL);
                checkResult(status, "createProgram clGetProgramInfo (binary size)");                    
                
                std::vector<char> binbuf(programBinarySize);
                const char *binbufptr = binbuf.data();
                status = clGetProgramInfo(program, CL_PROGRAM_BINARIES, sizeof(char*) * 1,
                                          &binbufptr, NULL);
                checkResult(status, "createProgram clGetProgramInfo (binary data)");                          
                
                std::ofstream ofs(binpath, std::ios::out | std::ios::binary);
                ofs.write(binbuf.data(), binbuf.size());
                ofs.close();
            }
            return 0;
        }             
        return buildProgram(dId);
    }


	int svc_SetUpOclObjects(cl_device_id dId, const std::string &kernel_code,
                            const std::string &kernel_name1, const std::string &kernel_name2,
                            const bool from_source, const bool save_binary, const bool reuse_binary) {

		cl_int status;
        const oclParameter *param = clEnvironment::instance()->getParameter(dId);
        assert(param);
        context    = param->context;
        cmd_queue  = param->commandQueue;

        if (!from_source) { 		//compile kernel on device
            if (buildKernelCode(kernel_code, dId)<0) return -1;
        } else {  // kernel_code is the path to the (binary?) source
            if (createProgram(kernel_code, dId, save_binary, reuse_binary)<0) return -1;
        }
        
        char buf[128];
        size_t s, ss[3];
        
        clGetDeviceInfo(dId, CL_DEVICE_NAME, 128, buf, NULL);
        clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &s, NULL);
        clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_ITEM_SIZES, 3*sizeof(size_t), &ss, NULL);
        //std::cerr << "svc_SetUpOclObjects dev " << dId << " " << buf << " MAX_WORK_GROUP_SIZE " << s << " MAX_WORK_ITEM_SIZES " << ss[0] << " " << ss[1] << " " << ss[2] << "\n";
        
        
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

        return 0;
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
    const bool from_source;
    bool my_own_allocator; 
    ff_oclallocator *allocator;
	const size_t width;
	const Tout identityVal;
	Tout reduceVar;

	cl_mem inputBuffer, outputBuffer;
    std::vector<std::pair<cl_mem, size_t> > envBuffer;

    /*
     * each accelerator works on the following subset of the input array:
     * [left-border][input-portion][right-border]
     * input portion is accessed RW, border are accessed read-only
     */
    size_t sizeInput; //byte-size of the input-portion
    size_t sizeInput_padded; //byte-size of the input-portion plus left and right borders
	size_t lenInput; //n. elements in the input-portion
	size_t offset1_in; //left-offset (begin input-portion wrt to begin input)
	size_t pad1_in; //n. elements in the left-border
	size_t pad2_in; //n. elements in the right-border
	size_t lenInput_global; //n. elements in the input

	size_t sizeOutput, sizeOutput_padded;
	size_t lenOutput, offset1_out, pad1_out, pad2_out, lenOutput_global;


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

    // build the program from the mapf and reducef functions
	ff_stencilReduceLoopOCL_1D(const std::string &mapf,			              //OpenCL elemental function 
                               const std::string &reducef = std::string(""),  //OpenCL combinator function
                               const Tout &identityVal = Tout(),               
                               ff_oclallocator *allocator = nullptr,
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(false), saveBinary(false), reuseBinary(false), 
        accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
        stencil_width(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL), 
        oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
		setcode(mapf, reducef);
        for(size_t i = 0; i< NACCELERATORS; ++i)
            accelerators[i]= new accelerator_t(allocator, width,identityVal);
	}
    
    // build the program from source code file, 
    // first attempts to load a cached binary file (kernels_source in this case is the path to the binary file)
    // ff that file is not available, then it creates the program from source and store the binary for future use
    // with the extention ".bin"
    ff_stencilReduceLoopOCL_1D(const std::string &kernels_source,            // OpenCL source code path
                               const std::string &mapf_name,                 // name of the map function
                               const std::string &reducef_name,              // name of the reduce function
                               const Tout &identityVal = Tout(),
                               ff_oclallocator *allocator = nullptr,
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(false), saveBinary(false), reuseBinary(false), 
        accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
        stencil_width(width), offset_dev(0), oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
		setsourcecode(kernels_source, mapf_name, reducef_name);
        for(size_t i = 0; i< NACCELERATORS; ++i)
            accelerators[i]= new accelerator_t(allocator, width, identityVal, true);
	}

    // the task is provided in the constructor -- one shot computation
	ff_stencilReduceLoopOCL_1D(const T &task, 
                               const std::string &mapf,	               
                               const std::string &reducef = std::string(""),	
                               const Tout &identityVal = Tout(),
                               ff_oclallocator *allocator = nullptr,
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(true), saveBinary(false), reuseBinary(false), 
        accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
        stencil_width(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL), 
        oldBytesizeIn(0), oldSizeOut(0), oldSizeReduce(0) {
		ff_node::skipfirstpop(true);
		setcode(mapf, reducef);
		Task.setTask(const_cast<T*>(&task));
        for(size_t i = 0; i< NACCELERATORS; ++i)
            accelerators[i]= new accelerator_t(allocator, width,identityVal);
	}

    // the task is provided in the constructor -- one shot computation
    ff_stencilReduceLoopOCL_1D(const T &task, 
                               const std::string &kernels_source,            // OpenCL source code path
                               const std::string &mapf_name,           // name of the map kernel function
                               const std::string &reducef_name,        // name of the reduce kernel function
                               const Tout &identityVal = Tout(),
                               ff_oclallocator *allocator = nullptr,
                               const int NACCELERATORS = 1, const int width = 1) :
        oneshot(true), saveBinary(false), reuseBinary(false), 
        accelerators(NACCELERATORS), acc_in(NACCELERATORS), acc_out(NACCELERATORS), 
        stencil_width(width), offset_dev(0), old_inPtr(NULL), old_outPtr(NULL), 
        oldBytesizeIn(0), oldSizeOut(0),  oldSizeReduce(0)  {
		setsourcecode(kernels_source, mapf_name, reducef_name);
        Task.setTask(const_cast<T*>(&task));
        for(size_t i = 0; i< NACCELERATORS; ++i)
            accelerators[i]= new accelerator_t(allocator, width, identityVal, true);
	}
   

	virtual ~ff_stencilReduceLoopOCL_1D() {
        for(size_t i = 0; i< accelerators.size(); ++i)
            if (accelerators[i]) delete accelerators[i];
    }

    // used to set tasks when in onshot mode
    void setTask(const T &task) {
        assert(oneshot);
        Task.resetTask();
        Task.setTask(&task);
    }

    /**
     * explicitly set the OpenCL devices to be used
     *
     * @param dev is the vector of devices (OpenCL Ids) to be used
     */
    void setDevices(std::vector<cl_device_id> &dev) {
//        if (dev.size() > accelerators.size()) {
//            error("ff_stencilReduceLoopOCL_1D::setDevices: Too many devices provided, please increase the number of logical accelerators\n");
//            return -1;
//        }
        devices = dev;
//        return 0;
    }

    // force execution on the CPU
    void pickCPU () {
        std::cerr << "STATUS: Execution on CPU requested\n";
        ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_CPU);
    }
    
    // force execution on the GPU - as many as requested by the co-allocation strategy
    void pickGPU (size_t offset=0 /* referred to global list of devices */) {
        std::cerr << "STATUS: Execution on GPU requested\n";
        offset_dev=offset; //TODO check numbering
        ff_oclNode_t<T>::setDeviceType(CL_DEVICE_TYPE_GPU);
    }

    // after the compilation and building phases, the OpenCL program will be saved as binary file 
    // this action takes effect only if the compilation is made with source file (i.e. not using macroes)
    void saveBinaryFile() { saveBinary = true; }

    // tells the run-time to re-use the binary file available
    void reuseBinaryFile() { reuseBinary = true; }
    
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
    
    /**
	 * Performs a static allocation of OpenCL devices.
	 * Priority is given to GPU devices, falling back to CPU devices
	 * if needed (e.g. no GPU devices).
	 * Currently it does not mix GPUs with CPU
	 */
	int nodeInit() {
		if (ff_oclNode_t<T>::oclId < 0) { //check if already initialized
			ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();
			if (devices.size() == 0) { // the user didn't set any specific device
				switch (ff_oclNode_t<T>::getDeviceType()) {
				case CL_DEVICE_TYPE_ALL:
				case CL_DEVICE_TYPE_GPU: {
					// Retrive multiple logical GPU devices (non-exclusive mode)
					std::vector<ssize_t> logdev =
							clEnvironment::instance()->coAllocateGPUDeviceRR(
									accelerators.size(),
									(offset_dev == 0) ? -1 : offset_dev);
					if (logdev.size() == 0) {
						//could not fulfill allocation request
						if(ff_oclNode_t<T>::getDeviceType() == CL_DEVICE_TYPE_GPU) {
							error("not enough GPUs found !\n");
							return -1;
						}
						//if user did not require require GPU devices, fallback to CPU
					} else {
						//convert retrieved logical devices into opencl Ids
						devices.clear();
						for (size_t i = 0; i < logdev.size(); ++i)
							devices.push_back(
									clEnvironment::instance()->getDevice(
											logdev[i]));
						for (size_t i = 0; i < devices.size(); ++i)
							if (accelerators[i]->init(devices[i], kernel_code,
                                                      kernel_name1, kernel_name2, saveBinary, reuseBinary) < 0)
								return -1;
						break;
					}
				}
				case CL_DEVICE_TYPE_CPU: {
					if (accelerators.size() > 1) {
						error(
								"Multiple (>1) virtual accelerators on CPU are requested. Not yet implemented.\n");
						return -1;
					} else {
						//Retrieve the CPU device
						devices.clear();
						devices.push_back(clEnvironment::instance()->getDevice( //convert to OpenCL Id
								clEnvironment::instance()->getCPUDevice())); //retrieve logical device
						if (accelerators[0]->init(devices[0], kernel_code,
                                                  kernel_name1, kernel_name2, saveBinary, reuseBinary) < 0)
							return -1;
					}
				}
				break;
				default: {
					error(
							"stencilReduceOCL::Other device. Not yet implemented.\n");
					return -1;
				}
				} //end switch on ff_oclNode_t<T>::getDeviceType()
			} else {
				//user requested specific OpenCL devices
				if (devices.size() > accelerators.size()) {
					error(
							"stencilReduceOCL::nodeInit: Too many devices requested, increase the number of accelerators!\n");
					return -1;
				}
				// NOTE: the number of devices requested can be lower than the number of accelerators.
				// TODO must be managed
				for (size_t i = 0; i < devices.size(); ++i)
					accelerators[i]->init(devices[i], kernel_code, kernel_name1,
                                          kernel_name2, saveBinary, reuseBinary);
			}
		}
        
        for (size_t i = 0; i < devices.size(); ++i)
            std::cerr << "Using " << clEnvironment::instance()->getDeviceInfo(devices[i]) << std::endl;

		return 0;
	}
    
    void nodeEnd() {
    	//TODO check:
    	// if multi-device, casuses multiple releaseAllBuffers calls to same object
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i]->releaseAll();
    }
     
protected:

	virtual bool isPureMap() const    { return false; }
	virtual bool isPureReduce() const { return false; }

    virtual int svc_init() { return nodeInit(); }

#if 0    
	virtual int svc_init() {
        if (ff_oclNode_t<T>::oclId < 0) {
            ff_oclNode_t<T>::oclId = clEnvironment::instance()->getOCLID();
            
            switch(ff_oclNode_t<T>::getDeviceType()) {
            case CL_DEVICE_TYPE_ALL:
                fprintf(stderr,"STATUS: requested ALL\n");
            case CL_DEVICE_TYPE_GPU: {// One or more GPUs
                // Not exclusive
                // Retrive logical devices
                std::vector<ssize_t> logdev = clEnvironment::instance()->coAllocateGPUDeviceRR(accelerators.size());
                // Convert into opencl Ids
                devices.clear();
                for (size_t i = 0; i < logdev.size(); ++i)
                    devices.push_back(clEnvironment::instance()->getDevice(logdev[i]));
                if (devices.size() == 0) {
                    error("stencilReduceOCL::svc_init:not enough GPUs found !\n");
                    return -1;
                } else {
                    // Ok
                    for (size_t i = 0; i < devices.size(); ++i)
                        accelerators[i]->init(devices[i], kernel_code, kernel_name1,kernel_name2);
                    break;
                }
            }
            case CL_DEVICE_TYPE_CPU: {
                if (accelerators.size()>1) {
                    error ("Multiple (>1) virtual accelerators on CPU are requested. Not yet implemented.\n");
                    return -1;
                } else {
                    // Ok
                    devices.clear();
                    devices.push_back(clEnvironment::instance()->getDevice(clEnvironment::instance()->getCPUDevice()));
                    accelerators[0]->init(devices[0], kernel_code, kernel_name1,kernel_name2);
                }
            } break;
            default: {
                error("stencilReduceOCL::Other device. Not yet implemented.\n");
            } break;
            }
        }
        return 0;
    }
#endif 

	virtual void svc_end() {
		if (!ff::ff_node::isfrozen()) nodeEnd();
	}

	T *svc(T *task) {
		if (task) {
            Task.resetTask();
            Task.setTask(task);
        }
		Tin   *inPtr     = Task.getInPtr();
		Tout  *outPtr    = Task.getOutPtr();
        Tout  *reducePtr = Task.getReduceVar();
        const size_t envSize = Task.getEnvNum(); //n. added environments
        
		// adjust allocator input-portions and relocate input device memory if needed
		if (oldBytesizeIn != Task.getBytesizeIn()) {
			compute_accmem(Task.getSizeIn(),  acc_in);

            for (size_t i = 0; i < accelerators.size(); ++i) {
                accelerators[i]->adjustInputBufferOffset(inPtr, old_inPtr, acc_in[i], Task.getSizeIn());
            }

            if (oldBytesizeIn < Task.getBytesizeIn()) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i]->relocateInputBuffer(inPtr, Task.getReuseIn(), reducePtr);
                }
                oldBytesizeIn = Task.getBytesizeIn();
            }            
		}

		// adjust allocator output-portions and relocate output device memory if needed
        if (oldSizeOut != Task.getBytesizeOut()) {

            if ((void*)inPtr == (void*)outPtr) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i]->setInPlace();
                }
            } else {
                compute_accmem(Task.getSizeOut(), acc_out);

                for (size_t i = 0; i < accelerators.size(); ++i) {
                    accelerators[i]->adjustOutputBufferOffset(outPtr, old_outPtr, acc_out[i], Task.getSizeOut());
                }
                
                if (oldSizeOut < Task.getBytesizeOut()) {
                    for (size_t i = 0; i < accelerators.size(); ++i) {
                        accelerators[i]->relocateOutputBuffer(outPtr); 
                    }
                    oldSizeOut = Task.getBytesizeOut();
                }
            }
        }

        //relocate env device memory
        //TODO on-demand relocate, as for input/output memory
        /* NOTE: env buffer are replicated on all devices.
         *       It would be nice to have replicated/partitioned polices
         */
        for (size_t i = 0; i < accelerators.size(); ++i)
            for(size_t k=0; k < envSize; ++k) {
                char *envptr;
                Task.getEnvPtr(k, envptr);
                accelerators[i]->relocateEnvBuffer(envptr, Task.getReuseEnv(k), k, Task.getBytesizeEnv(k));
            }        

		if (!isPureReduce())  //set kernel args
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i]->setMapKernelArgs(envSize);

		//(async) copy input and environments (h2d)
		for (size_t i = 0; i < accelerators.size(); ++i) {

            if (Task.getCopyIn())
                accelerators[i]->asyncH2Dinput(Task.getInPtr());  //in

            for(size_t k=0; k < envSize; ++k) {
                if (Task.getCopyEnv(k)) {
                    char *envptr;
                    Task.getEnvPtr(k, envptr);
                    accelerators[i]->asyncH2Denv(k, envptr);
                }
            }
        } 

		if (isPureReduce()) {
			//init reduce
			for (size_t i = 0; i < accelerators.size(); ++i)
                accelerators[i]->initReduce(REDUCE_INPUT);

			//wait for cross-accelerator h2d
			waitforh2d();

			//(async) device-reduce1
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i]->asyncExecReduceKernel1();

			//(async) device-reduce2
			for (size_t i = 0; i < accelerators.size(); ++i)
				accelerators[i]->asyncExecReduceKernel2();

			waitforreduce(); //wait for cross-accelerator reduce

			//host-reduce
			Tout redVar = accelerators[0]->getReduceVar();
			for (size_t i = 1; i < accelerators.size(); ++i)
				redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());
			Task.writeReduceVar(redVar);
		} else {
			Task.resetIter();

			if (isPureMap()) {
				//wait for cross-accelerator h2d
				waitforh2d();

				//(async) exec kernel
				for (size_t i = 0; i < accelerators.size(); ++i)
					accelerators[i]->asyncExecMapKernel();
				Task.incIter();

				waitformap(); //join

			} else { //iterative Map-Reduce (aka stencilReduceLoop)

                //invalidate first swap
				for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i]->swap();

				bool go = true;
				do {
					//Task.before();

					for (size_t i = 0; i < accelerators.size(); ++i)	accelerators[i]->swap();

					//wait for cross-accelerator h2d
					waitforh2d();

					//(async) execute MAP kernel
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i]->asyncExecMapKernel();
					Task.incIter();

					//start async-interleaved: reduce + borders sync
					//init reduce
					for (size_t i = 0; i < accelerators.size(); ++i)
                        accelerators[i]->initReduce();

					//(async) device-reduce1
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i]->asyncExecReduceKernel1();

					//(async) device-reduce2
					for (size_t i = 0; i < accelerators.size(); ++i)
						accelerators[i]->asyncExecReduceKernel2();

					//wait for cross-accelerators reduce
					waitforreduce();

					//host-reduce
					Tout redVar = accelerators[0]->getReduceVar();
					for (size_t i = 1; i < accelerators.size(); ++i) 
						redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());
                    Task.writeReduceVar(redVar);

					go = Task.iterCondition_aux();
					if (go) {
                        assert(outPtr);
						//(async) read back borders (d2h)
						for (size_t i = 0; i < accelerators.size(); ++i)
							accelerators[i]->asyncD2Hborders(outPtr);
						waitford2h(); //wait for cross-accelerators d2h
						//(async) read borders (h2d)
						for (size_t i = 0; i < accelerators.size(); ++i)
							accelerators[i]->asyncH2Dborders(outPtr);
					}

					//Task.after();

				} while (go);
			}
            
            //(async)read back output (d2h)
            if (outPtr && Task.getCopyOut()) { // do we have to copy back the output result ?
                for (size_t i = 0; i < accelerators.size(); ++i)
                    accelerators[i]->asyncD2Houtput(outPtr);
                waitford2h(); //wait for cross-accelerators d2h
            }
		}

        if (Task.getReleaseIn()) {
            for (size_t i = 0; i < accelerators.size(); ++i) 
                accelerators[i]->releaseInput(inPtr);
            oldBytesizeIn = 0;
            old_inPtr = NULL;
        } else old_inPtr = inPtr;

        if (Task.getReleaseOut()) {
            for (size_t i = 0; i < accelerators.size(); ++i) 
                accelerators[i]->releaseOutput(outPtr);
            oldSizeOut = 0;
            old_outPtr = NULL;
        } else old_outPtr = outPtr;

        for(size_t k=0; k < envSize; ++k) {
            if (Task.getReleaseEnv(k)) {
                for (size_t i = 0; i < accelerators.size(); ++i) {
                    char *envptr;
                    Task.getEnvPtr(k, envptr);
                    accelerators[i]->releaseEnv(k,envptr);
                }
            }

            // TODO: management of oldEnvPtr !!
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

	void setsourcecode(const std::string &source, const std::string &kernel1, const std::string &kernel2) {		
        if (kernel1 != "") kernel_name1 = "kern_"+kernel1;
        if (kernel2 != "") kernel_name2 = "kern_"+kernel2;
        kernel_code  = source;
    }

	//assign input partition to accelerators
	//acc[i] = (start, size) where:
	//         - start is the first element assigned to accelerator i
	//         - size is the number of elements assigned to accelerator i
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
			accelerators[i]->waitforh2d();
	}

	void waitford2h() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i]->waitford2h();
	}

	void waitforreduce() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i]->waitforreduce();
	}

	void waitformap() {
		for (size_t i = 0; i < accelerators.size(); ++i)
			accelerators[i]->waitformap();
	}

private:
	TOCL Task;
	const bool oneshot;
    bool saveBinary, reuseBinary;    
    std::vector<accelerator_t*> accelerators;
    std::vector<std::pair<size_t, size_t> > acc_in;
    std::vector<std::pair<size_t, size_t> > acc_out;
    std::vector<cl_device_id> devices;
	int stencil_width;
    //size_t preferred_dev;
    size_t offset_dev;

	std::string kernel_code;
	std::string kernel_name1;
	std::string kernel_name2;

    size_t forced_cpu;
    size_t forced_gpu;
    size_t forced_other;
    
    Tin   *old_inPtr;
    Tout  *old_outPtr;
	size_t oldBytesizeIn, oldSizeOut, oldSizeReduce;
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
	ff_mapOCL_1D(std::string mapf, ff_oclallocator *alloc=nullptr,
                 const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, "", 0, alloc, NACCELERATORS, 0) {
	}

    ff_mapOCL_1D(const std::string &kernels_source, const std::string &mapf_name, 
                 ff_oclallocator *alloc=nullptr, const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_source, mapf_name, "", 0, alloc, NACCELERATORS, 0) {
	}
    

	ff_mapOCL_1D(const T &task, std::string mapf, 
                 ff_oclallocator *alloc=nullptr,
                 const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, "", 0, alloc, NACCELERATORS, 0) {
	}
    ff_mapOCL_1D(const T &task, const std::string &kernels_source, const std::string &mapf_name, 
                 ff_oclallocator *alloc=nullptr,
                 const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_source, mapf_name, "", 0, alloc, NACCELERATORS, 0) {
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
                    ff_oclallocator *alloc=nullptr,
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>("", reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
	ff_reduceOCL_1D(const std::string &kernels_source, const std::string &reducef_name, const Tout &identityVal = Tout(),
                    ff_oclallocator *alloc=nullptr,
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_source, "", reducef_name, identityVal, alloc, NACCELERATORS, 0) {
	}

	ff_reduceOCL_1D(const T &task, std::string reducef, const Tout identityVal = Tout(),
                    ff_oclallocator *alloc=nullptr,
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, "", reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
	ff_reduceOCL_1D(const T &task, const std::string &kernels_source,const std::string &reducef_name, 
                    const Tout identityVal = Tout(),
                    ff_oclallocator *alloc=nullptr,
                    const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_source, "", reducef_name, identityVal, alloc, NACCELERATORS, 0) {
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
                       ff_oclallocator *alloc=nullptr,
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
	}

	ff_mapReduceOCL_1D(const std::string &kernels_code, const std::string &mapf_name, 
                       const std::string &reducef_name, const Tout &identityVal = Tout(),
                       ff_oclallocator *alloc=nullptr,
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(kernels_code, mapf_name, reducef_name, identityVal, alloc, NACCELERATORS, 0) {
	}


	ff_mapReduceOCL_1D(const T &task, std::string mapf, std::string reducef,
                       const Tout &identityVal = Tout(), 
                       ff_oclallocator *alloc=nullptr,
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, reducef, identityVal, alloc, NACCELERATORS, 0) {
	}
	ff_mapReduceOCL_1D(const T &task, const std::string &kernels_code, const std::string &mapf_name, 
                       const std::string &reducef_name, const Tout &identityVal = Tout(), 
                       ff_oclallocator *alloc=nullptr,
                       const size_t NACCELERATORS = 1) :
        ff_stencilReduceLoopOCL_1D<T, TOCL>(task, kernels_code, mapf_name, reducef_name, identityVal, alloc, NACCELERATORS, 0) {
	}

};

}
// namespace ff

#endif // FF_OCL
#endif /* FF_MAP_OCL_HPP */

