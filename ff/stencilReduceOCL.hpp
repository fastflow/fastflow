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
#include <ff/oclMacroes.hpp>

namespace ff {

// map base task for OpenCL implementation
// char type is a void type - clang++ don't like using void, g++ accepts it
//
// TODO: try to implement as a variadic template !!
//
template<typename TaskT_, typename Tin_, typename Tout_ = Tin_, typename Tenv1_ = char,
		typename Tenv2_ = char>
class baseOCLTask {
public:
	typedef TaskT_ TaskT;
	typedef Tin_ Tin;
	typedef Tout_ Tout;
	typedef Tenv1_ Tenv1;
	typedef Tenv2_ Tenv2;

	baseOCLTask() :
			inPtr(NULL), outPtr(NULL), envPtr1(NULL), envPtr2(NULL), size_in(0), size_out(
					0), size_env1(0), size_env2(0), reduceVar((Tout) 0), copyEnv1(
			true), copyEnv2(true), iter(0) {
	}

	virtual ~baseOCLTask() {
	}

	// user must override this method
	virtual void setTask(const TaskT *t) = 0;

	//user may override this methods
	virtual bool iterCondition(Tout, unsigned int) {
		return true;
	}

	bool iterCondition_aux() {
		return iterCondition(reduceVar, iter);
	}

	size_t getBytesizeIn() const {
		return getSizeIn() * sizeof(Tin);
	}
	size_t getBytesizeOut() const {
		return getSizeOut() * sizeof(Tout);
	}
	size_t getBytesizeEnv1() const {
		return getSizeEnv1() * sizeof(Tenv1);
	}
	size_t getBytesizeEnv2() const {
		return getSizeEnv2() * sizeof(Tenv2);
	}

	void setSizeIn(size_t sizeIn) {
		size_in = sizeIn;
	}
	void setSizeOut(size_t sizeOut) {
		size_out = sizeOut;
	}
	void setSizeEnv1(size_t sizeEnv) {
		size_env1 = sizeEnv;
	}
	void setSizeEnv2(size_t sizeEnv) {
		size_env2 = sizeEnv;
	}

	size_t getSizeIn() const {
		return size_in;
	}
	size_t getSizeOut() const {
		return (size_out == 0) ? size_in : size_out;
	}
	size_t getSizeEnv1() const {
		return size_env1;
	}
	size_t getSizeEnv2() const {
		return size_env2;
	}

	void setInPtr(Tin* _inPtr) {
		inPtr = _inPtr;
	}
	void setOutPtr(Tout* _outPtr) {
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
	Tout* getOutPtr() const {
		return outPtr;
	}
	Tenv1* getEnvPtr1() const {
		return envPtr1;
	}
	Tenv2* getEnvPtr2() const {
		return envPtr2;
	}

	void setReduceVar(const Tout r) {
		reduceVar = r;
	}

	Tout getReduceVar() const {
		return reduceVar;
	}

	Tout getInitReduceVal() {
		return initReduceVal;
	}

	void setInitReduceVal(Tout x) {
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
	Tout *outPtr;
	Tenv1 *envPtr1;
	Tenv2 *envPtr2;

	size_t size_in, size_out;
	size_t size_env1, size_env2;
	Tout reduceVar, initReduceVal;

	bool copyEnv1, copyEnv2;

	unsigned int iter;
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
class ff_ocl: public ff_oclNode_t<T> {
private:
	void setcode(const std::string &codestr1, const std::string &codestr2) {
		int n = codestr1.find_first_of("|");
		assert(n > 0);
		ff_ocl<T, TOCL>::kernel_name1 = codestr1.substr(0, n);
		const std::string &tmpstr = codestr1.substr(n + 1);
		n = tmpstr.find_first_of("|");
		assert(n > 0);

		// checking for double type
		if (tmpstr.substr(0, n) == "double") {
			kernel_code = "\n#pragma OPENCL EXTENSION cl_khr_fp64: enable\n\n"
					+ tmpstr.substr(n + 1);
		} else
			kernel_code = "\n" + tmpstr.substr(n + 1);

		// checking for extra code needed to compile the kernels
		std::ifstream ifs(FF_OPENCL_DATATYPES_FILE);
		if (ifs.is_open())
			kernel_code.insert(kernel_code.begin(), std::istreambuf_iterator<char>(ifs),
					std::istreambuf_iterator<char>());

		if (codestr2 != "") {
			n = codestr2.find("|");
			assert(n > 0);
			ff_ocl<T, TOCL>::kernel_name2 += codestr2.substr(0, n);
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

		kernel_code = "struct env_t {int x,y;};\n" + kernel_code;
	}
public:

	ff_ocl(const std::string &codestr1, const std::string &codestr2 = std::string("")) :
			oneshot(false) {
		setcode(codestr1, codestr2);
		oldSizeIn = oldSizeOut = oldSizeReduce = oldSizeEnv1 = oldSizeEnv2 = 0;
	}
	ff_ocl(const T &task, const std::string &codestr1, const std::string &codestr2 =
			std::string("")) :
			oneshot(true) {
		setcode(codestr1, codestr2);
		Task.setTask(const_cast<T*>(&task));
		oldSizeIn = oldSizeOut = oldSizeReduce = oldSizeEnv1 = oldSizeEnv2 = 0;
	}

//    void setKernel2() {
//        if (kernel) clReleaseKernel(kernel);
//        kernel = clCreateKernel(program, kernel_name2.c_str(), &status);
//        checkResult(status, "CreateKernel (2)");
//
//        status = clGetKernelWorkGroupInfo(kernel, ff_oclNode::baseclass_ocl_node_deviceId,
//                                  CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t), &workgroup_size,0);
//        checkResult(status, "GetKernelWorkGroupInfo");
//    }

	const T* getTask() const {
		return &Task;
	}
protected:
	inline void checkResult(cl_int s, const char* msg) {
		if (s != CL_SUCCESS) {
			std::cerr << msg << ":";
			printOCLErrorString(s, std::cerr);
		}
	}

	void buildKernelCode(const std::string &kc, cl_device_id dId) {
		cl_int status;

		size_t sourceSize = kc.length();
		const char* code = kc.c_str();

		//printf("code=\n%s\n", code);
		program = clCreateProgramWithSource(context, 1, &code, &sourceSize, &status);
		checkResult(status, "creating program with source");

		status = clBuildProgram(program, 1, &dId, "-cl-fast-relaxed-math", NULL, NULL);
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

	void svc_SetUpOclObjects(cl_device_id dId) {
		cl_int status;
		context = clCreateContext(NULL, 1, &dId, NULL, NULL, &status);
		checkResult(status, "creating context");

		cmd_queue = clCreateCommandQueue(context, dId, 0, &status);
		checkResult(status, "creating command-queue");

		//compile kernel on device
		buildKernelCode(kernel_code, dId);

		//create kernel objects
		kernel_map = clCreateKernel(program, kernel_name1.c_str(), &status);
		checkResult(status, "CreateKernel (map)");
		status = clGetKernelWorkGroupInfo(kernel_map, dId,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_map, 0);
		checkResult(status, "GetKernelWorkGroupInfo (map)");

		kernel_reduce = clCreateKernel(program, kernel_name2.c_str(), &status);
		checkResult(status, "CreateKernel (reduce)");
		status = clGetKernelWorkGroupInfo(kernel_reduce, dId,
		CL_KERNEL_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size_reduce, 0);
		checkResult(status, "GetKernelWorkGroupInfo (reduce)");

// allocate memory on device having the initial size
		if (Task.getBytesizeIn() > 0) {
			inputBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, Task.getBytesizeIn(),
					NULL, &status);
			checkResult(status, "CreateBuffer input (1)");
			oldSizeIn = Task.getBytesizeIn();
		}
	}

	void svc_releaseOclObjects() {
		if (ff_node::isfrozen())
			return;
		clReleaseKernel(kernel_map);
		clReleaseKernel(kernel_reduce);
		clReleaseProgram(program);
		clReleaseCommandQueue(cmd_queue);
		clReleaseMemObject(inputBuffer);
		if (oldSizeEnv1 > 0)
			clReleaseMemObject(envBuffer1);
		if (oldSizeEnv2 > 0)
			clReleaseMemObject(envBuffer2);
		if (oldSizeOut > 0)
			clReleaseMemObject(outputBuffer);
#if 0 //REDUCE
		if (oldSizeReduce > 0) {
			clReleaseMemObject(reduceBuffer);
			free(reduceMemInit);
		}
#endif
		clReleaseContext(context);
	}

	cl_mem* getInputBuffer() const {
		return (cl_mem*) &inputBuffer;
	}
	cl_mem* getOutputBuffer() const {
		return (cl_mem*) &outputBuffer;
	}
	cl_mem* getEnv1Buffer() const {
		return (cl_mem*) &envBuffer1;
	}
	cl_mem* getEnv2Buffer() const {
		return (cl_mem*) &envBuffer2;
	}
//	void swapInOut() {
//		inputBuffer = outputBuffer;
//	}
//	void swapBuffers() {
//		cl_mem tmp = inputBuffer;
//		inputBuffer = outputBuffer;
//		outputBuffer = tmp;
//	}

protected:
	const bool oneshot;
	TOCL Task;
	std::string kernel_code;
	std::string kernel_name1;
	std::string kernel_name2;
	size_t workgroup_size_map, workgroup_size_reduce;
	size_t oldSizeIn, oldSizeOut;
	size_t oldSizeReduce;
	size_t oldSizeEnv1, oldSizeEnv2;
	cl_context context;
	cl_program program;
	cl_command_queue cmd_queue;
	cl_mem inputBuffer;
	cl_mem outputBuffer;
	cl_mem reduceBuffer;
	cl_mem envBuffer1, envBuffer2;
#if 0 //REDUCE
	typename T::Tout *reduceMemInit;
#endif
	cl_kernel kernel_map, kernel_reduce;
	cl_int status;
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
class ff_stencilReduceOCL: public ff_ocl<T, TOCL> {
public:
	typedef typename TOCL::Tout Tout;

	ff_stencilReduceOCL(std::string mapf, std::string reducef, Tout initReduceVar =
			(Tout) 0) :
			ff_ocl<T, TOCL>(mapf, reducef) {
		ff_ocl<T, TOCL>::Task.setInitReduceVal(initReduceVar);
	}
	ff_stencilReduceOCL(const T &task, std::string mapf, std::string reducef,
			Tout initReduceVar = (Tout) 0) :
			ff_ocl<T, TOCL>(task, mapf, reducef) {
		ff_node::skipfirstpop(true);
		ff_ocl<T, TOCL>::Task.setInitReduceVal(initReduceVar);
	}
	virtual ~ff_stencilReduceOCL() {
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

	void setTask(const T &task) {
		ff_ocl<T, TOCL>::Task.setTask(&task);
	}
	const T* getTask() const {
		return ff_ocl<T, TOCL>::getTask();
	}

	double ffTime() {
		return ff_node::ffTime();
	}
	double ffwTime() {
		return ff_node::wffTime();
	}
	unsigned int getIter() {
		return ff_ocl<T, TOCL>::Task.getIter();
	}
	void setPureMap() {
		isPureMap = true;
	}
	void setPureReduce() {
		isPureReduce = true;
	}

protected:

	virtual bool isPureMap() {return false;}
	virtual bool isPureReduce() {return false;}

	inline void swapDeviceBuffers() {
		cl_mem tmp = ff_ocl<T, TOCL>::inputBuffer;
		ff_ocl<T, TOCL>::inputBuffer = ff_ocl<T, TOCL>::outputBuffer;
		ff_ocl<T, TOCL>::outputBuffer = tmp;
	}

	inline void checkResult(cl_int s, const char* msg) {
		if (s != CL_SUCCESS) {
			std::cerr << msg << ":";
			printOCLErrorString(s, std::cerr);
		}
	}

#if 0
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
#endif

	T *svc(T *task) {
		cl_int status = CL_SUCCESS;
		unsigned int nevents = 0;
		cl_event events[3];
		size_t globalThreads[2];
		size_t localThreads[2];

		if (task)
			ff_ocl<T, TOCL>::Task.setTask(task);
		size_t size = ff_ocl<T, TOCL>::Task.getSizeOut();
		const size_t inSize = ff_ocl<T, TOCL>::Task.getSizeIn();
		void* inPtr = ff_ocl<T, TOCL>::Task.getInPtr();
		void* outPtr = ff_ocl<T, TOCL>::Task.getOutPtr();
		void *envPtr1 = ff_ocl<T, TOCL>::Task.getEnvPtr1();
		void *envPtr2 = ff_ocl<T, TOCL>::Task.getEnvPtr2();

		//(eventually) allocate device memory
		if (ff_ocl<T, TOCL>::oldSizeIn < ff_ocl<T, TOCL>::Task.getBytesizeIn()) {
			if (ff_ocl<T, TOCL>::oldSizeIn != 0)
				clReleaseMemObject(ff_ocl<T, TOCL>::inputBuffer);
			ff_ocl<T, TOCL>::inputBuffer = clCreateBuffer(ff_ocl<T, TOCL>::context,
					CL_MEM_READ_WRITE, ff_ocl<T, TOCL>::Task.getBytesizeIn(), NULL,
					&status);
			ff_ocl<T, TOCL>::checkResult(status, "CreateBuffer input (2)");
			ff_ocl<T, TOCL>::oldSizeIn = ff_ocl<T, TOCL>::Task.getBytesizeIn();
		}

		if (inPtr == outPtr) {
			ff_ocl<T, TOCL>::outputBuffer = ff_ocl<T, TOCL>::inputBuffer;
		} else {
			if (ff_ocl<T, TOCL>::oldSizeOut < ff_ocl<T, TOCL>::Task.getBytesizeOut()) {
				if (ff_ocl<T, TOCL>::oldSizeOut != 0)
					clReleaseMemObject(ff_ocl<T, TOCL>::outputBuffer);

				ff_ocl<T, TOCL>::outputBuffer = clCreateBuffer(ff_ocl<T, TOCL>::context,
						CL_MEM_READ_WRITE, ff_ocl<T, TOCL>::Task.getBytesizeOut(), NULL,
						&status);
				ff_ocl<T, TOCL>::checkResult(status, "CreateBuffer output");
				ff_ocl<T, TOCL>::oldSizeOut = ff_ocl<T, TOCL>::Task.getBytesizeOut();
			}
		}
		if (envPtr1) {
			if (ff_ocl<T, TOCL>::oldSizeEnv1 < ff_ocl<T, TOCL>::Task.getBytesizeEnv1()) {
				if (ff_ocl<T, TOCL>::oldSizeEnv1 != 0)
					clReleaseMemObject(ff_ocl<T, TOCL>::envBuffer1);
				ff_ocl<T, TOCL>::envBuffer1 = clCreateBuffer(ff_ocl<T, TOCL>::context,
						CL_MEM_READ_ONLY, ff_ocl<T, TOCL>::Task.getBytesizeEnv1(), NULL,
						&status);
				ff_ocl<T, TOCL>::checkResult(status, "CreateBuffer env");
				ff_ocl<T, TOCL>::oldSizeEnv1 = ff_ocl<T, TOCL>::Task.getBytesizeEnv1();
			}
		}
		if (envPtr2) {
			if (ff_ocl<T, TOCL>::oldSizeEnv2 < ff_ocl<T, TOCL>::Task.getBytesizeEnv2()) {
				if (ff_ocl<T, TOCL>::oldSizeEnv2 != 0)
					clReleaseMemObject(ff_ocl<T, TOCL>::envBuffer2);
				ff_ocl<T, TOCL>::envBuffer2 = clCreateBuffer(ff_ocl<T, TOCL>::context,
						CL_MEM_READ_ONLY, ff_ocl<T, TOCL>::Task.getBytesizeEnv2(), NULL,
						&status);
				ff_ocl<T, TOCL>::checkResult(status, "CreateBuffer env2");
				ff_ocl<T, TOCL>::oldSizeEnv2 = ff_ocl<T, TOCL>::Task.getBytesizeEnv2();
			}
		}

		if (!isPureMap()) {
#if 0 //REDUCE
			//(eventually) allocate device memory for REDUCE - TODO check size
			size_t elemSize = sizeof(Tout);
			size_t numBlocks_reduce = 0;
			size_t numThreads_reduce = 0;
			getBlocksAndThreads(size, 64, 256, numBlocks_reduce, numThreads_reduce);// 64 and 256 are the max number of blocks and threads we want to use
			size_t reduceMemSize =
//				(numThreads_reduce <= 32) ?
//						(2 * numThreads_reduce * elemSize) :
//						(numThreads_reduce * elemSize);
			numBlocks_reduce * elemSize;
			if (ff_ocl<T, TOCL>::oldSizeReduce < reduceMemSize) {
				if (ff_ocl<T, TOCL>::oldSizeReduce != 0) {
					clReleaseMemObject(ff_ocl<T, TOCL>::reduceBuffer);
					free(ff_ocl<T, TOCL>::reduceMemInit);
				}
				ff_ocl<T, TOCL>::reduceBuffer = clCreateBuffer(ff_ocl<T, TOCL>::context,
						CL_MEM_READ_WRITE, numBlocks_reduce * elemSize, NULL, &status);
				ff_ocl<T, TOCL>::checkResult(status, "CreateBuffer reduce");
				ff_ocl<T, TOCL>::reduceMemInit = (Tout *) malloc(numBlocks_reduce * elemSize);
				for (size_t i = 0; i < numBlocks_reduce; ++i)
				ff_ocl<T, TOCL>::reduceMemInit[i] =
				ff_ocl<T, TOCL>::Task.getInitReduceVal();
				ff_ocl<T, TOCL>::oldSizeReduce = reduceMemSize;
			}
#endif
		}

		//configure MAP parallelism
		if (size < ff_ocl<T, TOCL>::workgroup_size_map) {
			localThreads[0] = size;
			globalThreads[0] = size;
		} else {
			localThreads[0] = ff_ocl<T, TOCL>::workgroup_size_map;
			globalThreads[0] = nextMultipleOfIf(size,
					ff_ocl<T, TOCL>::workgroup_size_map);
		}

		//set iteration-invariant MAP kernel args
		status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 2, sizeof(cl_uint),
				(void *) &inSize);
		ff_ocl<T, TOCL>::checkResult(status, "setKernelArg inSize");
		if (envPtr1) {
			if (envPtr2) {
				status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 3, sizeof(cl_mem),
						ff_ocl<T, TOCL>::getEnv1Buffer());
				ff_ocl<T, TOCL>::checkResult(status, "setKernelArg env1");
				status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 4, sizeof(cl_mem),
						ff_ocl<T, TOCL>::getEnv2Buffer());
				ff_ocl<T, TOCL>::checkResult(status, "setKernelArg env2");
				status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 5, sizeof(cl_uint),
						(void *) &size);
				ff_ocl<T, TOCL>::checkResult(status, "setKernelArg size");
			} else {
				status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 3, sizeof(cl_mem),
						ff_ocl<T, TOCL>::getEnv1Buffer());
				ff_ocl<T, TOCL>::checkResult(status, "setKernelArg env1");
				status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 4, sizeof(cl_uint),
						(void *) &size);
				ff_ocl<T, TOCL>::checkResult(status, "setKernelArg size");
			}
		} else {
			status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 3, sizeof(cl_uint),
					(void *) &size);
			ff_ocl<T, TOCL>::checkResult(status, "setKernelArg size");
		}

		//set iteration-dynamic MAP kernel args (init)
		status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 0, sizeof(cl_mem),
				ff_ocl<T, TOCL>::getInputBuffer());
		ff_ocl<T, TOCL>::checkResult(status, "setKernelArg input");
		status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 1, sizeof(cl_mem),
				ff_ocl<T, TOCL>::getOutputBuffer());
		ff_ocl<T, TOCL>::checkResult(status, "setKernvirtualelArg output");

		if (!isPureMap()) {
#if 0 //REDUCE
			//set iteration-invariant REDUCE kernel args
			status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 1, sizeof(cl_mem),
					&(ff_ocl<T, TOCL>::reduceBuffer));
			status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 2, sizeof(cl_uint),
					(void *) &size);
			status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 3, reduceMemSize,
					NULL);
			checkResult(status, "setKernelArg reduceMemSize");
#endif
		}

		//copy input and environments (h2d)
		status = clEnqueueWriteBuffer(ff_ocl<T, TOCL>::cmd_queue,
				ff_ocl<T, TOCL>::inputBuffer, CL_FALSE, 0,
				ff_ocl<T, TOCL>::Task.getBytesizeIn(), ff_ocl<T, TOCL>::Task.getInPtr(),
				0, NULL, &events[0]);
		++nevents;
		ff_ocl<T, TOCL>::checkResult(status, "copying Task to device input-buffer");
		if (envPtr1 && ff_ocl<T, TOCL>::Task.getCopyEnv1()) {
			status = clEnqueueWriteBuffer(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::envBuffer1, CL_FALSE, 0,
					ff_ocl<T, TOCL>::Task.getBytesizeEnv1(),
					ff_ocl<T, TOCL>::Task.getEnvPtr1(), 0, NULL, &events[1]);
			ff_ocl<T, TOCL>::checkResult(status, "copying Task to device env 1 buffer");
			++nevents;
		}
		if (envPtr2 && ff_ocl<T, TOCL>::Task.getCopyEnv2()) {
			status = clEnqueueWriteBuffer(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::envBuffer2, CL_FALSE, 0,
					ff_ocl<T, TOCL>::Task.getBytesizeEnv2(),
					ff_ocl<T, TOCL>::Task.getEnvPtr2(), 0, NULL, &events[2]);
			ff_ocl<T, TOCL>::checkResult(status, "copying Task to device env2 buffer");
			++nevents;
		}
		clWaitForEvents(nevents, events);
		nevents = 0;

		if (isPureReduce()) {
#if 0 //REDUCE
			//begin REDUCE
			//init REDUCE memory - TODO parallel
			status = clEnqueueWriteBuffer(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::reduceBuffer, CL_FALSE, 0, reduceMemSize,
					ff_ocl<T, TOCL>::reduceMemInit, 0, NULL, NULL);
			ff_ocl<T, TOCL>::checkResult(status, "init Reduce buffer");

			localThreads[0] = numThreads_reduce;
			globalThreads[0] = numBlocks_reduce * numThreads_reduce;

			//set iteration-dynamic REDUCE kernel args
			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 0, sizeof(cl_mem),
					ff_ocl<T, TOCL>::getOutputBuffer());

			//execute first REDUCE kernel (output to blocks)
			status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::kernel_reduce, 1, NULL, globalThreads, localThreads,
					0, NULL, &events[0]);
			status = clWaitForEvents(1, &events[0]);

			// Sets the kernel arguments for second REDUCE
			size = numBlocks_reduce;
			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 0, sizeof(cl_mem),
					&(ff_ocl<T, TOCL>::reduceBuffer));
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 1, sizeof(cl_mem),
//					&(ff_ocl<T, TOCL>::reduceBuffer));
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 2, sizeof(cl_uint),
//					(void*) &size);
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 3, reduceMemSize,
//			NULL);
			ff_ocl<T, TOCL>::checkResult(status, "setKernelArg ");

			localThreads[0] = numThreads_reduce;
			globalThreads[0] = numThreads_reduce;

			//execute second REDUCE kernel (blocks to value)
			status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::kernel_reduce, 1, NULL, globalThreads, localThreads,
					0, NULL, &events[0]);

			//read back REDUCE var (d2h)
			Tout reduceVar;
			status |= clWaitForEvents(1, &events[0]);
			status |= clEnqueueReadBuffer(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::reduceBuffer, CL_TRUE, 0, elemSize, &reduceVar, 0,
					NULL, &events[1]);
			status |= clWaitForEvents(1, &events[1]);
			ff_ocl<T, TOCL>::checkResult(status, "ERROR during OpenCL computation");
			ff_ocl<T, TOCL>::Task.setReduceVar(reduceVar);
			//end REDUCE
#endif
		}

		else {
			ff_ocl<T, TOCL>::Task.resetIter();

			if (isPureMap()) {
				//execute MAP kernel
				status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
						ff_ocl<T, TOCL>::kernel_map, 1, NULL, globalThreads, localThreads,
						0, NULL, &events[0]);
				ff_ocl<T, TOCL>::Task.incIter();
				status |= clWaitForEvents(1, &events[0]);
			}

			else { //iterative Map-Reduce (aka stencilReduce)
				do {

					//execute MAP kernel
					status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
							ff_ocl<T, TOCL>::kernel_map, 1, NULL, globalThreads,
							localThreads, 0, NULL, &events[0]);

					ff_ocl<T, TOCL>::Task.incIter();
					swapDeviceBuffers();

					//set iteration-dynamic MAP kernel args
					status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 0,
							sizeof(cl_mem), ff_ocl<T, TOCL>::getInputBuffer());
					ff_ocl<T, TOCL>::checkResult(status, "setKernelArg input");
					status = clSetKernelArg(ff_ocl<T, TOCL>::kernel_map, 1,
							sizeof(cl_mem), ff_ocl<T, TOCL>::getOutputBuffer());
					ff_ocl<T, TOCL>::checkResult(status, "setKernelArg output");

					status |= clWaitForEvents(1, &events[0]);
					//end MAP

#if 0 //REDUCE
					//begin REDUCE
					//init REDUCE memory - TODO parallel
					status = clEnqueueWriteBuffer(ff_ocl<T, TOCL>::cmd_queue,
							ff_ocl<T, TOCL>::reduceBuffer, CL_FALSE, 0, reduceMemSize,
							ff_ocl<T, TOCL>::reduceMemInit, 0, NULL, NULL);
					ff_ocl<T, TOCL>::checkResult(status, "init Reduce buffer");

					localThreads[0] = numThreads_reduce;
					globalThreads[0] = numBlocks_reduce * numThreads_reduce;

					//set iteration-dynamic REDUCE kernel args
					status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 0, sizeof(cl_mem),
							ff_ocl<T, TOCL>::getOutputBuffer());

					//execute first REDUCE kernel (output to blocks)
					status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
							ff_ocl<T, TOCL>::kernel_reduce, 1, NULL, globalThreads, localThreads,
							0, NULL, &events[0]);
					status = clWaitForEvents(1, &events[0]);

					// Sets the kernel arguments for second REDUCE
					size = numBlocks_reduce;
					status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 0, sizeof(cl_mem),
							&(ff_ocl<T, TOCL>::reduceBuffer));
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 1, sizeof(cl_mem),
//					&(ff_ocl<T, TOCL>::reduceBuffer));
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 2, sizeof(cl_uint),
//					(void*) &size);
//			status |= clSetKernelArg(ff_ocl<T, TOCL>::kernel_reduce, 3, reduceMemSize,
//			NULL);
					ff_ocl<T, TOCL>::checkResult(status, "setKernelArg ");

					localThreads[0] = numThreads_reduce;
					globalThreads[0] = numThreads_reduce;

					//execute second REDUCE kernel (blocks to value)
					status = clEnqueueNDRangeKernel(ff_ocl<T, TOCL>::cmd_queue,
							ff_ocl<T, TOCL>::kernel_reduce, 1, NULL, globalThreads, localThreads,
							0, NULL, &events[0]);

					//read back REDUCE var (d2h)
					Tout reduceVar;
					status |= clWaitForEvents(1, &events[0]);
					status |= clEnqueueReadBuffer(ff_ocl<T, TOCL>::cmd_queue,
							ff_ocl<T, TOCL>::reduceBuffer, CL_TRUE, 0, elemSize, &reduceVar, 0,
							NULL, &events[1]);
					status |= clWaitForEvents(1, &events[1]);
					ff_ocl<T, TOCL>::checkResult(status, "ERROR during OpenCL computation");
					ff_ocl<T, TOCL>::Task.setReduceVar(reduceVar);
					//end REDUCE
#endif
				} while (ff_ocl<T, TOCL>::Task.iterCondition_aux());
			}

			//read back output (d2h)
			status |= clEnqueueReadBuffer(ff_ocl<T, TOCL>::cmd_queue,
					ff_ocl<T, TOCL>::outputBuffer, CL_TRUE, 0,
					ff_ocl<T, TOCL>::Task.getBytesizeOut(),
					ff_ocl<T, TOCL>::Task.getOutPtr(), 0, NULL, NULL);
		}

		//TODO check
//		for(int i=0; i<3; ++i)
//			if(events[i]) clReleaseEvent(events[i]);

		return (ff_ocl<T, TOCL>::oneshot ? NULL : task);
	}
}
;

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
class ff_mapOCL: public ff_stencilReduceOCL<T, TOCL> {
public:
	ff_mapOCL(std::string mapf) :
			ff_stencilReduceOCL<T, TOCL>(mapf, "") {
	}

	ff_mapOCL(const T &task, std::string mapf) :
			ff_stencilReduceOCL<T, TOCL>(task, mapf, "") {
	}
	bool isPureMap() {return true;}
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
class ff_reduceOCL: public ff_stencilReduceOCL<T, TOCL> {
public:
	typedef typename TOCL::Tout Tout;

	ff_reduceOCL(std::string reducef, Tout initReduceVar = (Tout) 0) :
			ff_stencilReduceOCL<T, TOCL>("", reducef, initReduceVar) {
	}
	ff_reduceOCL(const T &task, std::string reducef, Tout initReduceVar = (Tout) 0) :
			ff_stencilReduceOCL<T, TOCL>(task, "", reducef, initReduceVar) {
	}
	bool isPureReduce() {return true;}
};

#if 0
/*
 * \class ff_mapreduceOCL
 *  \ingroup high_level_patterns_shared_memory
 *
 * \brief The map reduce skeleton using OpenCL
 *
 *
 */
template<typename T, typename TOCL=T>
class ff_mapreduceOCL: public ff_ocl<T,TOCL> {
public:
	typedef typename TOCL::Tout Tout;

	ff_mapreduceOCL(std::string mapf, std::string reducef):ff_ocl<T,TOCL>(mapf,reducef) {}
	ff_mapreduceOCL(const T &task, std::string mapf, std::string reducef):
	ff_ocl<T,TOCL>(task, mapf, reducef) {
		ff_node::skipfirstpop(true);
	}
	virtual ~ff_mapreduceOCL() {}

	virtual int run(bool=false) {return ff_node::run();}
	virtual int wait() {return ff_node::wait();}

	virtual int run_and_wait_end() {
		if (run()<0) return -1;
		if (wait()<0) return -1;
		return 0;
	}

	virtual int run_then_freeze() {
		return ff_node::freeze_and_run();
	}
	virtual int wait_freezing() {
		return ff_node::wait_freezing();
	}

	void setTask(const T &task) {ff_ocl<T,TOCL>::Task.setTask(&task);}
	const T* getTask() const {return ff_ocl<T,TOCL>::getTask();}

	double ffTime() {return ff_node::ffTime();}
	double ffwTime() {return ff_node::wffTime();}

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
		const size_t half = (size+1)/2;
		threads = (size < maxThreads*2) ?
		( isPowerOf2(half) ? nextPowerOf2(half+1) : nextPowerOf2(half) ) : maxThreads;
		blocks = (size + (threads * 2 - 1)) / (threads * 2);
		blocks = std::min(maxBlocks, blocks);
	}

	T *svc(T *task) {
		cl_int status = CL_SUCCESS;
		cl_event events[2];
		size_t globalThreads[1];
		size_t localThreads[1];

		if (task) ff_ocl<T,TOCL>::Task.setTask(task);
		size_t size = ff_ocl<T,TOCL>::Task.getSizeOut();
		const size_t inSize = ff_ocl<T,TOCL>::Task.getSizeIn();
		void* inPtr = ff_ocl<T,TOCL>::Task.getInPtr();
		void* outPtr = ff_ocl<T,TOCL>::Task.getOutPtr();
		void *envPtr1 = ff_ocl<T,TOCL>::Task.getEnvPtr1();
		void *envPtr2 = ff_ocl<T,TOCL>::Task.getEnvPtr2();

		// MAP
		if (size < ff_ocl<T,TOCL>::workgroup_size) {
			localThreads[0] = size;
			globalThreads[0] = size;
		} else {
			localThreads[0] = ff_ocl<T,TOCL>::workgroup_size;
			globalThreads[0] = nextMultipleOfIf(size,ff_ocl<T,TOCL>::workgroup_size);
		}

		if ( ff_ocl<T,TOCL>::oldSizeIn < ff_ocl<T,TOCL>::Task.getBytesizeIn() ) {
			if (ff_ocl<T,TOCL>::oldSizeIn != 0) clReleaseMemObject(ff_ocl<T,TOCL>::inputBuffer);
			ff_ocl<T,TOCL>::inputBuffer = clCreateBuffer(ff_ocl<T,TOCL>::context, CL_MEM_READ_WRITE,
					ff_ocl<T,TOCL>::Task.getBytesizeIn(), NULL, &status);
			ff_ocl<T,TOCL>::checkResult(status, "CreateBuffer input (2)");
			ff_ocl<T,TOCL>::oldSizeIn = ff_ocl<T,TOCL>::Task.getBytesizeIn();
		}

		if ( inPtr == outPtr ) {
			ff_ocl<T,TOCL>::outputBuffer = ff_ocl<T,TOCL>::inputBuffer;
		} else {
			if ( ff_ocl<T,TOCL>::oldSizeOut < ff_ocl<T,TOCL>::Task.getBytesizeOut() ) {
				if (ff_ocl<T,TOCL>::oldSizeOut != 0) clReleaseMemObject(ff_ocl<T,TOCL>::outputBuffer);

				ff_ocl<T,TOCL>::outputBuffer = clCreateBuffer(ff_ocl<T,TOCL>::context, CL_MEM_READ_WRITE,
						ff_ocl<T,TOCL>::Task.getBytesizeOut(), NULL, &status);
				ff_ocl<T,TOCL>::checkResult(status, "CreateBuffer output");
				ff_ocl<T,TOCL>::oldSizeOut = ff_ocl<T,TOCL>::Task.getBytesizeOut();
			}
		}

		if (envPtr1) {
			if (ff_ocl<T,TOCL>::oldSizeEnv1 < ff_ocl<T,TOCL>::Task.getBytesizeEnv1()) {
				if (ff_ocl<T,TOCL>::oldSizeEnv1 != 0) clReleaseMemObject(ff_ocl<T,TOCL>::envBuffer1);
				ff_ocl<T,TOCL>::envBuffer1 = clCreateBuffer(ff_ocl<T,TOCL>::context, CL_MEM_READ_ONLY,
						ff_ocl<T,TOCL>::Task.getBytesizeEnv1(), NULL, &status);
				ff_ocl<T,TOCL>::checkResult(status, "CreateBuffer env");
				ff_ocl<T,TOCL>::oldSizeEnv1 = ff_ocl<T,TOCL>::Task.getBytesizeEnv1();
			}
		}
		if (envPtr2) {
			if (ff_ocl<T,TOCL>::oldSizeEnv2 < ff_ocl<T,TOCL>::Task.getBytesizeEnv2()) {
				if (ff_ocl<T,TOCL>::oldSizeEnv2 != 0) clReleaseMemObject(ff_ocl<T,TOCL>::envBuffer2);
				ff_ocl<T,TOCL>::envBuffer2 = clCreateBuffer(ff_ocl<T,TOCL>::context, CL_MEM_READ_ONLY,
						ff_ocl<T,TOCL>::Task.getBytesizeEnv2(), NULL, &status);
				ff_ocl<T,TOCL>::checkResult(status, "CreateBuffer env2");
				ff_ocl<T,TOCL>::oldSizeEnv2 = ff_ocl<T,TOCL>::Task.getBytesizeEnv2();
			}
		}

		status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 0, sizeof(cl_mem), ff_ocl<T,TOCL>::getInputBuffer());
		ff_ocl<T,TOCL>::checkResult(status, "setKernelArg input");
		status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 1, sizeof(cl_mem), ff_ocl<T,TOCL>::getOutputBuffer());
		ff_ocl<T,TOCL>::checkResult(status, "setKernelArg output");
		status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 2, sizeof(cl_uint), (void *)&inSize);
		ff_ocl<T,TOCL>::checkResult(status, "setKernelArg size");
		if (envPtr1) {
			if (envPtr2) {
				status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 3, sizeof(cl_mem), ff_ocl<T,TOCL>::getEnv1Buffer());
				ff_ocl<T,TOCL>::checkResult(status, "setKernelArg env1");
				status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 4, sizeof(cl_mem), ff_ocl<T,TOCL>::getEnv2Buffer());
				ff_ocl<T,TOCL>::checkResult(status, "setKernelArg env2");
				status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 5, sizeof(cl_uint), (void *)&size);
				ff_ocl<T,TOCL>::checkResult(status, "setKernelArg size");
			} else {
				status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 3, sizeof(cl_mem), ff_ocl<T,TOCL>::getEnv1Buffer());
				ff_ocl<T,TOCL>::checkResult(status, "setKernelArg env1");
				status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 4, sizeof(cl_uint), (void *)&size);
				ff_ocl<T,TOCL>::checkResult(status, "setKernelArg size");
			}
		} else {
			status = clSetKernelArg(ff_ocl<T,TOCL>::kernel, 3, sizeof(cl_uint), (void *)&size);
			ff_ocl<T,TOCL>::checkResult(status, "setKernelArg size");
		}

		status = clEnqueueWriteBuffer(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::inputBuffer,CL_FALSE,0,
				ff_ocl<T,TOCL>::Task.getBytesizeIn(), ff_ocl<T,TOCL>::Task.getInPtr(),
				0,NULL,NULL);
		ff_ocl<T,TOCL>::checkResult(status, "copying Task to device input-buffer");
		if ( envPtr1 && ff_ocl<T,TOCL>::Task.getCopyEnv1() ) {
			status = clEnqueueWriteBuffer(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::envBuffer1,CL_FALSE,0,
					ff_ocl<T,TOCL>::Task.getBytesizeEnv1(), ff_ocl<T,TOCL>::Task.getEnvPtr1(),
					0,NULL,NULL);
			ff_ocl<T,TOCL>::checkResult(status, "copying Task to device env 1 buffer");
		}
		if ( envPtr2 && ff_ocl<T,TOCL>::Task.getCopyEnv2() ) {
			status = clEnqueueWriteBuffer(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::envBuffer2,CL_FALSE,0,
					ff_ocl<T,TOCL>::Task.getBytesizeEnv2(), ff_ocl<T,TOCL>::Task.getEnvPtr2(),
					0,NULL,NULL);
			ff_ocl<T,TOCL>::checkResult(status, "copying Task to device env2 buffer");
		}

		status = clEnqueueNDRangeKernel(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::kernel,1,NULL,
				globalThreads, localThreads,0,NULL,&events[0]);

		status |= clWaitForEvents(1, &events[0]);

		// REDUCE
		ff_ocl<T,TOCL>::setKernel2();

		size_t elemSize = sizeof(Tout);
		size_t numBlocks = 0;
		size_t numThreads = 0;

		// 64 and 256 are the max number of blocks and threads we want to use
		getBlocksAndThreads(size, 64, 256, numBlocks, numThreads);

		size_t outMemSize =
		(numThreads <= 32) ? (2 * numThreads * elemSize) : (numThreads * elemSize);

		localThreads[0] = numThreads;
		globalThreads[0] = numBlocks * numThreads;

		// input buffer is the output buffer from map stage
		ff_ocl<T,TOCL>::swapInOut();

		ff_ocl<T,TOCL>::outputBuffer = clCreateBuffer(ff_ocl<T,TOCL>::context, CL_MEM_READ_WRITE,numBlocks*elemSize, NULL, &status);
		ff_ocl<T,TOCL>::checkResult(status, "CreateBuffer output");

		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 0, sizeof(cl_mem), ff_ocl<T,TOCL>::getInputBuffer());
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 1, sizeof(cl_mem), ff_ocl<T,TOCL>::getOutputBuffer());
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 2, sizeof(cl_uint), (void *)&size);
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 3, outMemSize, NULL);
		checkResult(status, "setKernelArg ");

		status = clEnqueueNDRangeKernel(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::kernel,1,NULL,
				globalThreads,localThreads,0,NULL,&events[0]);
		status = clWaitForEvents(1, &events[0]);

		// Sets the kernel arguments for second reduction
		size = numBlocks;
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 0, sizeof(cl_mem),ff_ocl<T,TOCL>::getOutputBuffer());
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 1, sizeof(cl_mem),ff_ocl<T,TOCL>::getOutputBuffer());
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 2, sizeof(cl_uint),(void*)&size);
		status |= clSetKernelArg(ff_ocl<T,TOCL>::kernel, 3, outMemSize, NULL);
		ff_ocl<T,TOCL>::checkResult(status, "setKernelArg ");

		localThreads[0] = numThreads;
		globalThreads[0] = numThreads;

		status = clEnqueueNDRangeKernel(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::kernel,1,NULL,
				globalThreads,localThreads,0,NULL,&events[0]);
		Tout *reduceVar = ff_ocl<T,TOCL>::Task.getReduceVar();
		assert(reduceVar != NULL);
		status |= clWaitForEvents(1, &events[0]);
		status |= clEnqueueReadBuffer(ff_ocl<T,TOCL>::cmd_queue,ff_ocl<T,TOCL>::outputBuffer,CL_TRUE, 0,
				elemSize, reduceVar ,0,NULL,&events[1]);
		status |= clWaitForEvents(1, &events[1]);
		ff_ocl<T,TOCL>::checkResult(status, "ERROR during OpenCL computation");

		clReleaseEvent(events[0]);
		clReleaseEvent(events[1]);

		//return (ff_ocl<T,TOCL>::oneshot?NULL:task);
		return (ff_ocl<T,TOCL>::oneshot?NULL:task);
	}
};
#endif

}
// namespace ff

#endif // FF_OCL

#endif /* FF_MAP_OCL_HPP */

