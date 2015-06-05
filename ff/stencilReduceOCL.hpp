/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file stecilReduceOCL.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief OpenCL map and non-iterative data-parallel patterns
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

#ifndef FF_STENCILREDUCE_OCL_HPP
#define FF_STENCILREDUCE_OCL_HPP

#ifdef FF_OCL

#include <string>
#include <fstream>
#include <ff/oclnode.hpp>
#include <ff/node.hpp>
#include <ff/mapper.hpp>
#include <ff/stencilReduceOCL_macros.hpp>

namespace ff {

// map base task for OpenCL implementation
// char type is a void type - clang++ don't like using void, g++ accepts it
//
// TODO: try to implement as a variadic template !!
//
template<typename TaskT_, typename Tin_, typename Tenv1_ = char, typename Tenv2_ = char>
class baseOCLTask {
public:
	typedef TaskT_ TaskT;
	typedef Tin_ Tin;
	typedef Tenv1_ Tenv1;
	typedef Tenv2_ Tenv2;

	baseOCLTask() :
			inPtr(NULL), outPtr(NULL), envPtr1(NULL), envPtr2(NULL), size_in(0), copyEnv1(
					true), copyEnv2(true), iter(0), reduceVar(NULL), initReduceVal(
					(Tin) 0) {
	}

	virtual ~baseOCLTask() {
	}

	// user must override this method
	virtual void setTask(const TaskT *t) = 0;

//		//user may overrider this methods
	virtual bool iterCondition(Tin, unsigned int) {
		return false;
	} //default: map (1 iter)
	virtual Tin combinator(Tin, Tin) {
		return (Tin) 0;
	}
	;
//		virtual void before() {} //the very first op of each iter
//		virtual void after() {} //the very last op of each iter

	bool iterCondition_aux() {
		return iterCondition(*reduceVar, iter);
	}

	size_t getBytesizeIn() const {
		return getSizeIn() * sizeof(Tin);
	}

	void setSizeIn(size_t sizeIn) {
		size_in = sizeIn;
	}

	size_t getSizeIn() const {
		return size_in;
	}

	void setInPtr(Tin* _inPtr) {
		inPtr = _inPtr;
	}
	void setOutPtr(Tin* _outPtr) {
		outPtr = _outPtr;
	}
	void setEnvPtr1(const Tenv1* _envPtr) {
		envPtr1 = (Tenv1*) _envPtr;
	}
	void setEnvPtr2(const Tenv2* _envPtr) {
		envPtr2 = (Tenv2*) _envPtr;
	}

	bool getCopyEnv1() const {
		return copyEnv1;
	}
	bool getCopyEnv2() const {
		return copyEnv2;
	}

	void setCopyEnv1(bool c) {
		copyEnv1 = c;
	}
	void setCopyEnv2(bool c) {
		copyEnv2 = c;
	}

	Tin* getInPtr() const {
		return inPtr;
	}
	Tin* getOutPtr() const {
		return outPtr;
	}
	Tenv1* getEnvPtr1() const {
		return envPtr1;
	}
	Tenv2* getEnvPtr2() const {
		return envPtr2;
	}

	void setReduceVar(const Tin *r) {
		reduceVar = (Tin *) r;
	}

	void writeReduceVar(const Tin r) {
		*reduceVar = r;
	}

//		Tin getReduceVar() const {
//			return reduceVar_;
//		}

	Tin getInitReduceVal() {
		return initReduceVal;
	}

	void setInitReduceVal(Tin x) {
		initReduceVal = x;
	}

	void incIter() {
		++iter;
	}
	unsigned int getIter() {
		return iter;
	}

	void resetIter() {
		iter = 0;
	}

protected:
	Tin *inPtr;
	Tin *outPtr;
	Tenv1 *envPtr1;
	Tenv2 *envPtr2;

	size_t size_in;

	bool copyEnv1, copyEnv2;

	unsigned int iter;

private:
	Tin *reduceVar, initReduceVal;
};

enum reduceMode {
	REDUCE_INPUT, REDUCE_OUTPUT
};

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
template<typename T, typename TOCL = T>
class ff_oclAccelerator {
public:
	typedef typename TOCL::Tin Tin;
	typedef typename TOCL::Tenv1 Tenv1;
	typedef typename TOCL::Tenv2 Tenv2;

	ff_oclAccelerator(const size_t width_, Tin initReduceVal_) :
			width(width_), initReduceVal(initReduceVal_), baseclass_ocl_node_deviceId(
			NULL) {
		workgroup_size_map = workgroup_size_reduce = 0;
		inputBuffer = outputBuffer = envBuffer1 = envBuffer2 = reduceBuffer = NULL;
		sizeInput = sizeInput_padded = sizeEnv1_padded = 0;
		lenInput = offset1 = offset2 = pad1 = pad2 = lenInput_global = 0;
		nevents_h2d = nevents_map = 0;
		event_d2h = event_map = event_reduce1 = event_reduce2 = NULL;
		localThreadsMap = globalThreadsMap = 0;
		localThreadsReduce = globalThreadsReduce = numBlocksReduce = blockMemSize = 0;
		reduceVar = initReduceVal;
		kernel_map = kernel_reduce = kernel_init = NULL;
		context = NULL;
		program = NULL;
		cmd_queue = NULL;
	}

	void init(int dtype, const std::string &kernel_code, const std::string &kernel_name1,
			const std::string &kernel_name2) {
		switch (dtype) {
		case CL_DEVICE_TYPE_GPU:
			baseclass_ocl_node_deviceId = threadMapper::instance()->getOCLgpu();
			break;
		case CL_DEVICE_TYPE_CPU:
			baseclass_ocl_node_deviceId = threadMapper::instance()->getOCLcpu();
			break;
		case CL_DEVICE_TYPE_ACCELERATOR:
			baseclass_ocl_node_deviceId = threadMapper::instance()->getOCLaccelerator();
			break;
		}
		svc_SetUpOclObjects(baseclass_ocl_node_deviceId, kernel_code, kernel_name1,
				kernel_name2);
	}

	void release() {
		svc_releaseOclObjects();
	}

	void swapBuffers() {
		cl_mem tmp = inputBuffer;
		inputBuffer = outputBuffer;
		outputBuffer = tmp;
	}

	void relocateInputBuffer(unsigned long len, unsigned long len_global,
			unsigned long first) {
		offset1 = first;
		lenInput = len;
		lenInput_global = len_global;
		pad1 = (std::min)(width, offset1);
		offset2 = lenInput_global - lenInput - offset1;
		pad2 = (std::min)(width, offset2);
		sizeInput = lenInput * sizeof(Tin);
		sizeInput_padded = sizeInput + (pad1 + pad2) * sizeof(Tin);
		sizeEnv1_padded = (lenInput + pad1 + pad2) * sizeof(Tenv1);
		cl_int status;
		if (inputBuffer)
			clReleaseMemObject(inputBuffer);
		//allocate input-size + pre/post-windows
		inputBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE, sizeInput_padded, NULL, &status);
		checkResult(status, "CreateBuffer input");
		if (lenInput < workgroup_size_map) {
			localThreadsMap = lenInput;
			globalThreadsMap = lenInput;
		} else {
			localThreadsMap = workgroup_size_map;
			globalThreadsMap = nextMultipleOfIf(lenInput, workgroup_size_map);
		}
		//REDUCE
		if (reduceBuffer)
			clReleaseMemObject(reduceBuffer);
		// 64 and 256 are the max number of blocks and threads we want to use
		getBlocksAndThreads(lenInput, 64, 256, numBlocksReduce, localThreadsReduce);
		globalThreadsReduce = numBlocksReduce * localThreadsReduce;

		blockMemSize =
				(localThreadsReduce <= 32) ?
						(2 * localThreadsReduce * sizeof(Tin)) :
						(localThreadsReduce * sizeof(Tin));

		reduceBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE,
				numBlocksReduce * sizeof(Tin), NULL, &status);
		checkResult(status, "CreateBuffer reduce");
	}

	void relocateOutputBuffer() {
		cl_int status;
		if (outputBuffer)
			clReleaseMemObject(outputBuffer);
		outputBuffer = clCreateBuffer(context,
		CL_MEM_READ_WRITE, sizeInput_padded, NULL, &status);
		checkResult(status, "CreateBuffer output");
	}

	void relocateEnvBuffer1() {
		cl_int status;
		if (envBuffer1)
			clReleaseMemObject(envBuffer1);
		envBuffer1 = clCreateBuffer(context,
		CL_MEM_READ_WRITE, sizeEnv1_padded, NULL, &status);
		checkResult(status, "CreateBuffer envBuffer1");
	}

	void relocateEnvBuffer2() {
		cl_int status;
		if (envBuffer2)
			clReleaseMemObject(envBuffer2);
		envBuffer2 = clCreateBuffer(context,
		CL_MEM_READ_WRITE, sizeof(Tenv2), NULL, &status);
		checkResult(status, "CreateBuffer envBuffer2");
	}

	void setInPlace() {
		outputBuffer = inputBuffer;
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

	void setMapKernelArgs() {
		cl_uint idx = 0;
		//set iteration-dynamic MAP kernel args (init)
		cl_int status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &inputBuffer);
		checkResult(status, "setKernelArg input");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &outputBuffer);
		checkResult(status, "setKernelArg output");
		//set iteration-invariant MAP kernel args
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint),
				(void *) &lenInput_global);
		checkResult(status, "setKernelArg global input length");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &envBuffer1);
		checkResult(status, "setKernelArg env1");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_mem), &envBuffer2);
		checkResult(status, "setKernelArg env2");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &lenInput);
		checkResult(status, "setKernelArg local input length");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &offset1);
		checkResult(status, "setKernelArg offset");
		status = clSetKernelArg(kernel_map, idx++, sizeof(cl_uint), (void *) &pad1);
		checkResult(status, "setKernelArg pad");
	}

	void asyncH2Dinput(Tin *p) {
		p += offset1 - pad1;
		cl_int status = clEnqueueWriteBuffer(cmd_queue, inputBuffer, CL_FALSE, 0,
				sizeInput_padded, p, 0, NULL, &events_h2d[nevents_h2d++]);
		checkResult(status, "copying Task to device input-buffer");
	}

	void asyncH2Denv1(Tenv1 *p) {
		p += offset1 - pad1;
		cl_int status = clEnqueueWriteBuffer(cmd_queue, envBuffer1, CL_FALSE, 0,
				sizeEnv1_padded, p, 0, NULL, &events_h2d[nevents_h2d++]);
		checkResult(status, "copying Task to device env1-buffer");
	}

	void asyncH2Denv2(Tenv2 *p) {
		cl_int status = clEnqueueWriteBuffer(cmd_queue, envBuffer2, CL_FALSE, 0,
				sizeof(Tenv2), p, 0, NULL, &events_h2d[nevents_h2d++]);
		checkResult(status, "copying Task to device env2-buffer");
	}

	void asyncD2Houtput(Tin *p) {
		cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
				pad1 * sizeof(Tin), sizeInput, p + offset1, 0, NULL, &event_d2h);
		checkResult(status, "copying output back from device");
	}

	void asyncD2Hborders(Tin *p) {
		cl_int status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
				pad1 * sizeof(Tin), width * sizeof(Tin), p + offset1, 0, NULL,
				&events_h2d[0]);
		checkResult(status, "copying border1 back from device");
		++nevents_h2d;
		status = clEnqueueReadBuffer(cmd_queue, outputBuffer, CL_FALSE,
				(pad1 + lenInput - width) * sizeof(Tin), width * sizeof(Tin),
				p + offset1 + lenInput - width, 0, NULL, &event_d2h);
		checkResult(status, "copying border2 back from device");
	}

	void asyncH2Dborders(Tin *p) {
		if (pad1) {
			cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE, 0,
					pad1 * sizeof(Tin), p + offset1 - pad1, 0, NULL,
					&events_h2d[nevents_h2d++]);
			checkResult(status, "copying left border to device");
		}
		if (pad2) {
			cl_int status = clEnqueueWriteBuffer(cmd_queue, outputBuffer, CL_FALSE,
					(pad1 + lenInput) * sizeof(Tin), pad2 * sizeof(Tin),
					p + offset1 + lenInput, 0, NULL, &events_h2d[nevents_h2d++]);
			checkResult(status, "copying right border to device");
		}
	}

	void initReduce(Tin initReduceVal, reduceMode m = REDUCE_OUTPUT) {
		//set kernel args for reduce1
		int idx = 0;
		cl_mem tmp = (m == REDUCE_OUTPUT) ? outputBuffer : inputBuffer;
		cl_int status = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &tmp);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &pad1);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),
				(void *) &lenInput);
		status |= clSetKernelArg(kernel_reduce, idx++, blockMemSize, NULL);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tin),
				(void *) &initReduceVal);
		checkResult(status, "setKernelArg reduce-1");
	}

	void asyncExecMapKernel() {
		//execute MAP kernel
		cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_map, 1, NULL,
				&globalThreadsMap, &localThreadsMap, 0, NULL, &event_map);
		checkResult(status, "executing map kernel");
		++nevents_map;
	}

	void asyncExecReduceKernel1() {
		cl_int status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
				&globalThreadsReduce, &localThreadsReduce, nevents_map, &event_map,
				&event_reduce1);
		checkResult(status, "exec kernel reduce-1");
		nevents_map = 0;
	}

	void asyncExecReduceKernel2() {
		cl_uint zeropad = 0;
		int idx = 0;
		cl_int status = clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem),
				&reduceBuffer);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(uint), &zeropad);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_mem), &reduceBuffer);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(cl_uint),
				(void*) &numBlocksReduce);
		status |= clSetKernelArg(kernel_reduce, idx++, blockMemSize, NULL);
		status |= clSetKernelArg(kernel_reduce, idx++, sizeof(Tin),
				(void *) &initReduceVal);
		checkResult(status, "setKernelArg reduce-2");

		status = clEnqueueNDRangeKernel(cmd_queue, kernel_reduce, 1, NULL,
				&localThreadsReduce, &localThreadsReduce, 1, &event_reduce1,
				&event_reduce2);
		checkResult(status, "exec kernel reduce-2");
	}

	Tin getReduceVar() {
		cl_int status = clEnqueueReadBuffer(cmd_queue, reduceBuffer, CL_TRUE, 0,
				sizeof(Tin), &reduceVar, 0, NULL, NULL);
		checkResult(status, "d2h reduceVar");
		return reduceVar;
	}

	void waitforh2d() {
		clWaitForEvents(nevents_h2d, events_h2d);
		nevents_h2d = 0;
	}

	void waitford2h() {
		clWaitForEvents(1, &event_d2h);
	}

	void waitforreduce() {
		clWaitForEvents(1, &event_reduce2);
	}

	void waitformap() {
		clWaitForEvents(nevents_map, &event_map);
		nevents_map = 0;
	}

public:
	cl_context context;
	cl_program program;
	cl_command_queue cmd_queue;
	cl_mem reduceBuffer;
	cl_kernel kernel_map, kernel_reduce, kernel_init;

private:
	void buildKernelCode(const std::string &kc, cl_device_id dId) {
		cl_int status;

		size_t sourceSize = kc.length();
		const char* code = kc.c_str();

		//printf("code=\n%s\n", code);
		program = clCreateProgramWithSource(context, 1, &code, &sourceSize, &status);
		checkResult(status, "creating program with source");

		status = clBuildProgram(program, 1, &dId, /*"-cl-fast-relaxed-math"*/NULL, NULL,
		NULL);
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
		context = clCreateContext(NULL, 1, &dId, NULL, NULL, &status);
		checkResult(status, "creating context");

		cmd_queue = clCreateCommandQueue(context, dId, 0, &status);
		checkResult(status, "creating command-queue");

		//compile kernel on device
		buildKernelCode(kernel_code, dId);

		//create kernel objects
		if (kernel_name1 != "") {
			kernel_map = clCreateKernel(program, kernel_name1.c_str(), &status);
			checkResult(status, "CreateKernel (map)");
			status = clGetKernelWorkGroupInfo(kernel_map, dId,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_map, 0);
			checkResult(status, "GetKernelWorkGroupInfo (map)");
		}

		if (kernel_name2 != "") {
			kernel_reduce = clCreateKernel(program, kernel_name2.c_str(), &status);
			checkResult(status, "CreateKernel (reduce)");
			status = clGetKernelWorkGroupInfo(kernel_reduce, dId,
			CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_reduce, 0);
			checkResult(status, "GetKernelWorkGroupInfo (reduce)");
		}
	}

	void svc_releaseOclObjects() {
		if (kernel_map)
			clReleaseKernel(kernel_map);
		if (kernel_reduce)
			clReleaseKernel(kernel_reduce);
		clReleaseProgram(program);
		clReleaseCommandQueue(cmd_queue);
		if (inputBuffer)
			clReleaseMemObject(inputBuffer);
		if (envBuffer1)
			clReleaseMemObject(envBuffer1);
		if (envBuffer2)
			clReleaseMemObject(envBuffer2);
		if (outputBuffer && outputBuffer != inputBuffer)
			clReleaseMemObject(outputBuffer);
		if (reduceBuffer)
			clReleaseMemObject(reduceBuffer);
		clReleaseContext(context);
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

	cl_mem inputBuffer, outputBuffer, envBuffer1, envBuffer2;
	const size_t width;
	size_t sizeInput, sizeInput_padded, sizeEnv1_padded;
	unsigned long lenInput, offset1, offset2, pad1, pad2, lenInput_global;
	size_t workgroup_size_map, workgroup_size_reduce;
	unsigned int nevents_h2d, nevents_map;
	cl_event events_h2d[3], event_d2h, event_map, event_reduce1, event_reduce2;
	size_t localThreadsMap, globalThreadsMap;
	size_t localThreadsReduce, globalThreadsReduce, numBlocksReduce, blockMemSize;
	Tin reduceVar;
	const Tin initReduceVal;
	cl_device_id baseclass_ocl_node_deviceId;	// is the id which is provided for user
};

/*
 * \class ff_mapreduceOCL
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The map reduce skeleton using OpenCL
 *
 *
 */
template<typename T, typename TOCL = T>
class ff_stencilReduceLoopOCL_1D: public ff_node_t<T> {
public:
	typedef typename TOCL::Tin Tin;
	typedef typename TOCL::Tenv1 Tenv1;
	typedef typename TOCL::Tenv2 Tenv2;
	typedef ff_oclAccelerator<T, TOCL> accelerator_t;

	//template <typename Function>
	ff_stencilReduceLoopOCL_1D(const std::string &mapf,			//OCL elemental
			const std::string &reducef = std::string(""),			//OCL combinator
			Tin initReduceVar = (Tin) 0, const int NACCELERATORS_ = 1, const int width_ =
					1) :
			oneshot(false), NACCELERATORS(NACCELERATORS_), stencil_width(width_), oldSizeIn(
					0), oldSizeReduce(0) {
		setcode(mapf, reducef);
		Task.setInitReduceVal(initReduceVar);
		accelerators = (accelerator_t **) malloc(NACCELERATORS * sizeof(accelerator_t *));
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i] = new accelerator_t(stencil_width, initReduceVar);
		acc_len = new size_t[NACCELERATORS];
		acc_off = new size_t[NACCELERATORS];
	}

	//template <typename Function>
	ff_stencilReduceLoopOCL_1D(const T &task, const std::string &mapf,	//OCL elemental
			const std::string &reducef = std::string(""),			//OCL combinator
			Tin initReduceVar = (Tin) 0, const int NACCELERATORS_ = 1, const int width_ =
					1) :
			oneshot(true), NACCELERATORS(NACCELERATORS_), stencil_width(width_), oldSizeIn(
					0), oldSizeReduce(0) {
		ff_node::skipfirstpop(true);
		setcode(mapf, reducef);
		Task.setTask(&task);
		Task.setInitReduceVal(initReduceVar);
		accelerators = (accelerator_t **) malloc(NACCELERATORS * sizeof(accelerator_t *));
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i] = new accelerator_t(stencil_width, initReduceVar);
		acc_len = new size_t[NACCELERATORS];
		acc_off = new size_t[NACCELERATORS];
	}

	virtual ~ff_stencilReduceLoopOCL_1D() {
		for (int i = 0; i < NACCELERATORS; ++i)
			delete accelerators[i];
		free(accelerators);
		delete[] acc_len;
		delete[] acc_off;
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

	Tin getReduceVar() {
		assert(oneshot);
		return Task.getReduceVar();
	}

protected:

	virtual bool isPureMap() {
		return false;
	}
	virtual bool isPureReduce() {
		return false;
	}

//		inline void checkResult(cl_int s, const char* msg) {
//			if (s != CL_SUCCESS) {
//				std::cerr << msg << ":";
//				printOCLErrorString(s, std::cerr);
//			}
//		}

	virtual int svc_init() {
		for (int i = 0; i < NACCELERATORS; ++i) {
			accelerators[i]->init(CL_DEVICE_TYPE_GPU, kernel_code, kernel_name1,
					kernel_name2);
		}
		return 0;
	}

	virtual void svc_end() {
		if (!ff::ff_node::isfrozen())
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->release();
	}

	T *svc(T *task) {
		if (task)
			Task.setTask(task);
		Tin* inPtr = Task.getInPtr();
		Tin* outPtr = Task.getOutPtr();
		Tenv1 *envPtr1 = Task.getEnvPtr1();
		Tenv2 *envPtr2 = Task.getEnvPtr2();

		//(eventually) relocate device memory
		if (oldSizeIn < Task.getBytesizeIn()) {
			compute_accmem(Task.getSizeIn());
			for (int i = 0; i < NACCELERATORS; ++i) {
				//input
				accelerators[i]->relocateInputBuffer(acc_len[i], Task.getSizeIn(),
						acc_off[i]);
				//output
				if (inPtr == outPtr)
					accelerators[i]->setInPlace();
				else
					accelerators[i]->relocateOutputBuffer();
				//env1
				if (envPtr1)
					accelerators[i]->relocateEnvBuffer1();
				//env2
				if (envPtr2)
					accelerators[i]->relocateEnvBuffer2();
			}
			oldSizeIn = Task.getBytesizeIn();
		}

		if (!isPureReduce())
			//set kernel args
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->setMapKernelArgs();

		//(async) copy input and environments (h2d)
		for (int i = 0; i < NACCELERATORS; ++i) {
			//in
			accelerators[i]->asyncH2Dinput(Task.getInPtr());
			//env1
			if (envPtr1 && Task.getCopyEnv1())
				accelerators[i]->asyncH2Denv1(Task.getEnvPtr1());
			//env2
			if (envPtr2 && Task.getCopyEnv2())
				accelerators[i]->asyncH2Denv2(Task.getEnvPtr2());
		}
//		//join
//		for (int i = 0; i < NACCELERATORS; ++i)
//			accelerators[i]->join();

		if (isPureReduce()) {
			//init reduce
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->initReduce(Task.getInitReduceVal(), REDUCE_INPUT);

			//wait for cross-accelerator h2d
			waitforh2d();

			//(async) device-reduce1
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->asyncExecReduceKernel1();

			//(async) device-reduce2
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->asyncExecReduceKernel2();

			waitforreduce(); //wait for cross-accelerator reduce

			//host-reduce
			Tin redVar = Task.getInitReduceVal();
			for (int i = 0; i < NACCELERATORS; ++i)
				redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());

			Task.writeReduceVar(redVar);
		}

		else {
			Task.resetIter();

			if (isPureMap()) {
				//wait for cross-accelerator h2d
				waitforh2d();

				//(async) exec kernel
				for (int i = 0; i < NACCELERATORS; ++i)
					accelerators[i]->asyncExecMapKernel();
				Task.incIter();

				//join
				waitformap();
			}

			else { //iterative Map-Reduce (aka stencilReduce)
				   //invalidate first swap
				for (int i = 0; i < NACCELERATORS; ++i)
					accelerators[i]->swap();
				bool go = true;
				do {

					//Task.before();

					for (int i = 0; i < NACCELERATORS; ++i)
						accelerators[i]->swap();

					//wait for cross-accelerator h2d
					waitforh2d();

					//(async) execute MAP kernel
					for (int i = 0; i < NACCELERATORS; ++i)
						accelerators[i]->asyncExecMapKernel();
					Task.incIter();

					//start async-interleaved: reduce + borders sync
					//init reduce
					for (int i = 0; i < NACCELERATORS; ++i)
						accelerators[i]->initReduce(Task.getInitReduceVal());

					//(async) device-reduce1
					for (int i = 0; i < NACCELERATORS; ++i)
						accelerators[i]->asyncExecReduceKernel1();

					//(async) device-reduce2
					for (int i = 0; i < NACCELERATORS; ++i)
						accelerators[i]->asyncExecReduceKernel2();

					//wait for cross-accelerators reduce
					waitforreduce();

					//host-reduce
					Tin redVar = Task.getInitReduceVal();
					for (int i = 0; i < NACCELERATORS; ++i)
						redVar = Task.combinator(redVar, accelerators[i]->getReduceVar());

					Task.writeReduceVar(redVar);

					go = Task.iterCondition_aux();
					if (go) {
						//(async) read back borders (d2h)
						for (int i = 0; i < NACCELERATORS; ++i)
							accelerators[i]->asyncD2Hborders(Task.getOutPtr());
						waitford2h(); //wait for cross-accelerators d2h
						//(async) read borders (h2d)
						for (int i = 0; i < NACCELERATORS; ++i)
							accelerators[i]->asyncH2Dborders(Task.getOutPtr());
					}

					//Task.after();

				} while (go);
			}

			//(async)read back output (d2h)
			for (int i = 0; i < NACCELERATORS; ++i)
				accelerators[i]->asyncD2Houtput(Task.getOutPtr());
			waitford2h(); //wait for cross-accelerators d2h
		}

		return (oneshot ? NULL : task);
	}

	TOCL Task;
	const bool oneshot;

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

	void compute_accmem(size_t len) {
		size_t start = 0, step = (len + NACCELERATORS - 1) / NACCELERATORS;
		int i = 0;
		for (; i < NACCELERATORS - 1; ++i) {
			acc_off[i] = start;
			acc_len[i] = step;
			start += step;
		}
		acc_off[i] = start;
		acc_len[i] = len - start;
	}

	void waitforh2d() {
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i]->waitforh2d();
	}

	void waitford2h() {
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i]->waitford2h();
	}

	void waitforreduce() {
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i]->waitforreduce();
	}

	void waitformap() {
		for (int i = 0; i < NACCELERATORS; ++i)
			accelerators[i]->waitformap();
	}

//	void waitforreduce() {
//			for(int i=0; i<NACCELERATORS; ++i)
//				accelerators[i]->waitforh2d();
//		}

	int NACCELERATORS, stencil_width;
	accelerator_t **accelerators;
	size_t *acc_len, *acc_off;

	std::string kernel_code;
	std::string kernel_name1;
	std::string kernel_name2;

	size_t oldSizeIn;
	size_t oldSizeReduce;
}
;

/*
 * \class ff_mapReduceOCL
 *  \ingroup high_level_patterns_shared_memory
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
	typedef typename TOCL::Tin Tin;

	ff_mapReduceOCL_1D(std::string mapf, std::string reducef, const Tin initReduceVar =
			(Tin) 0, const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, reducef, initReduceVar,
					NACCELERATORS, 0) {
	}

	ff_mapReduceOCL_1D(const T &task, std::string mapf, std::string reducef,
			const Tin initReduceVar = (Tin) 0, const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, reducef, initReduceVar,
					NACCELERATORS, 0) {
	}
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
template<typename T, typename TOCL = T>
class ff_mapOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
	ff_mapOCL_1D(std::string mapf, const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>(mapf, "", 0, NACCELERATORS, 0) {
	}

	ff_mapOCL_1D(const T &task, std::string mapf, const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>(task, mapf, "", 0, NACCELERATORS, 0) {
	}
	bool isPureMap() {
		return true;
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
template<typename T, typename TOCL = T>
class ff_reduceOCL_1D: public ff_stencilReduceLoopOCL_1D<T, TOCL> {
public:
	typedef typename TOCL::Tin Tin;

	ff_reduceOCL_1D(std::string reducef, Tin initReduceVar = (Tin) 0,
			const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>("", reducef, initReduceVar, NACCELERATORS,
					0) {
	}
	ff_reduceOCL_1D(const T &task, std::string reducef, Tin initReduceVar = (Tin) 0,
			const size_t NACCELERATORS = 1) :
			ff_stencilReduceLoopOCL_1D<T, TOCL>(task, "", reducef, initReduceVar,
					NACCELERATORS, 0) {
	}
	bool isPureReduce() {
		return true;
	}
};

}
// namespace ff

#endif // FF_OCL
#endif /* FF_MAP_OCL_HPP */

