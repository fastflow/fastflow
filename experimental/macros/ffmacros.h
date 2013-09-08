/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */
/* Author: Massimo Torquati <torquati@di.unipi.it>
 *
 * Proof-Of-Concept 
 */



#ifndef __FF_MACROES_HPP_
#define __FF_MACROES_HPP_

#include <vector>
#include <string>
#include <ff/node.hpp>
#include <ff/dnode.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/map.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;


/*++++++++++ type macroes +++++++++++ */


#define FFType_1(ntype, type, name)					\
struct ntype {		    					        \
    type name;							        \
    ntype() {}							        \
    ntype(type name):name(name) {}					\
    void prepareOUT(svector<iovec>& v,int& nnull) {			\
        struct iovec iov1={&name,sizeof(type)};				\
        v.push_back(iov1);						\
	nnull=0;							\
    }									\
    void unmarshall(svector<zmqTransport::msg_t*>* const v[],		\
		    const int vlen) {					\
	name = *(type*)((v[0]->operator[](0))->getData());		\
	delete (zmqTransport::msg_t*)(v[0]->operator[](0));		\
        delete v[0];							\
    }									\
    void cleanup() {}							\
}

// used for non array types
#define FFType_2(ntype, type1, name1, type2, name2)			\
struct ntype {							        \
    type1 name1;							\
    type2 name2;							\
    ntype() {}								\
    ntype(type1 name1, type2 name2):name1(name1),name2(name2) {}	\
    void prepareOUT(svector<iovec>& v,int& nnull) {			\
        struct iovec iov1={&name1,sizeof(type1)};			\
        v.push_back(iov1);						\
	struct iovec iov2={&name2,sizeof(type2)};			\
        v.push_back(iov2);						\
	nnull=1;							\
    }									\
    void unmarshall(svector<zmqTransport::msg_t*>* const v[],		\
		    const int vlen) {					\
	name1 = *(type1*)((v[0]->operator[](0))->getData());		\
	name2 = *(type2*)((v[0]->operator[](1))->getData());		\
	delete (zmqTransport::msg_t*)(v[0]->operator[](0));		\
	delete (zmqTransport::msg_t*)(v[0]->operator[](1));		\
        delete v[0];							\
    }									\
    void cleanup() {}							\
}

#define FFType_ptr_1(ntype, type1, name1, size1)			\
struct ntype {							        \
    type1 name1;							\
    const unsigned sz;							\
    zmqTransport::msg_t* msg1;						\
    ntype(const unsigned sz=size1):sz(sz),msg1(NULL) {}			\
    ntype(type1 name1,const unsigned sz=size1):				\
          name1(name1),sz(sz),msg1(NULL) {}				\
    void prepareOUT(svector<iovec>& v, int& nnull) {			\
      struct iovec iov1={name1,sz};					\
        v.push_back(iov1);						\
	nnull = 0;							\
    }									\
    void unmarshall(svector<zmqTransport::msg_t*>* const v[],		\
		    const int vlen) {					\
	msg1 = v[0]->operator[](0);					\
	name1 = (type1)(msg1->getData());				\
        delete v[0];							\
    }									\
									\
    void cleanup() {							\
	if (msg1) delete msg1;						\
    }									\
}

#define FFType_ptr_2(ntype, type1, name1, size1,type2, name2,size2)	\
struct ntype {							        \
    type1 name1;							\
    type2 name2;							\
    unsigned sz1;							\
    unsigned sz2;							\
    enum {SIZE1=size1, SIZE2=size2};					\
    zmqTransport::msg_t* msg1;						\
    zmqTransport::msg_t* msg2;						\
    ntype():sz1(size1),sz2(size2),msg1(NULL),msg2(NULL) {}		\
    ntype(type1 name1, type2 name2):name1(name1),name2(name2),		\
      sz1(size1),sz2(size2),msg1(NULL),msg2(NULL) {}			\
    void prepareOUT(svector<iovec>& v, int& nnull) {			\
      struct iovec iov1={name1,sz1};					\
      v.push_back(iov1);						\
      struct iovec iov2={name2,sz2};					\
      v.push_back(iov2);						\
      nnull = 1;							\
    }									\
    void unmarshall(svector<zmqTransport::msg_t*>* const v[],		\
		    const int vlen) {					\
        msg1  = v[0]->operator[](0);					\
	msg2  = v[0]->operator[](1);					\
	name1 = (type1)(msg1->getData());				\
	name2 = (type2)(msg2->getData());				\
        delete v[0];							\
    }									\
    void cleanup() {							\
	if (msg1) delete msg1;						\
	if (msg2) delete msg2;						\
    }									\
}




/* ++++++++++ distributed vs non-distributed version ++++++++++ */
#if !defined(FFDISTRIBUTED)
#define FFDISTRIBUTED 1
#endif
#if (FFDISTRIBUTED != 1) 
#define FFD_INIT_RECEIVER(COMM_IN)
#define FFD_INIT_SENDER(COMM_OUT)
#else
#define FFD_INIT_RECEIVER(COMM_IN)					\
    ff_dnode<COMM_IN>::init(name, address, nHosts, transp,		\
			    RECEIVER, transp->getProcId())

#define FFD_INIT_SENDER(COMM_OUT)					\
    ff_dnode<COMM_OUT>::init(name, address, nHosts, transp, SENDER,	\
			     transp->getProcId(),callback)


#endif

/* counting tasks */
#if defined(COUNT_TASKS)
#define DO_COUNT_TASKS        ++_cnt;
#define PRINT_COUNT_TASKS     printf("%ld computed tasks hostId=%d\n",_cnt,transp->getProcId());
#define COUNT_TASKS_DEFINE    unsigned long _cnt;
#define COUNT_TASKS_INIT      _cnt=0;
#else
#define DO_COUNT_TASKS        
#define PRINT_COUNT_TASKS     
#define COUNT_TASKS_DEFINE    
#define COUNT_TASKS_INIT
#endif

#if defined(OCL)
#include <ff/ff_oclnode.hpp>

/*++++++++++ oclnode  macroes   +++++++++++ */
static inline void checkResult(cl_int s, const char* msg) {
    if (s != CL_SUCCESS) {
        std::cerr << msg << ":";
        printOCLErrorString(s,std::cerr);
    }
}

#define stringfy(x) #x
#define KERNEL(MAPF) " __kernel void kern( __global float* input, __global float* output,  const  uint max_items) {  int i = get_global_id(0); if (i<max_items)" stringfy(MAPF) "}"


#define GPUMAPDEF1(name, mapfunc, basictype)			        \
class gpu##name: public ff_oclNode {				        \
public:									\
    gpu##name(basictype* Vin, basictype* Vout, size_t size):		\
    Vin(Vin),Vout(Vout), size(size) {}					\
                                                                        \
  void svc_SetUpOclObjects(cl_device_id dId) {	                        \
      printf("setup called\n");						\
    cl_int status;							\
    context = clCreateContext(NULL,1,&dId,NULL,NULL,&status);		\
    checkResult(status, "creating context");				\
    									\
    cmd_queue = clCreateCommandQueue (context, dId, 0, &status);	\
    checkResult(status, "creating command-queue");			\
    									\
    const char * source =  KERNEL(mapfunc(input,output,i));		\
    size_t sourceSize = strlen(source);					\
    									\
    program = clCreateProgramWithSource(context,1,			\
			(const char **)&source,&sourceSize,&status);    \
    checkResult(status, "creating program with source");		\
    									\
    status = clBuildProgram(program,1,&dId,NULL,NULL,NULL);		\
    checkResult(status, "building program");				\
    									\
    kernel = clCreateKernel(program, "kern", &status);			\
    checkResult(status, "CreateKernel");				\
    									\
    inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY,		\
			 sizeof(cl_float ) * size, NULL, &status);      \
    checkResult(status, "CreateBuffer");				\
    									\
    outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY,		\
			  sizeof(cl_float ) * size, NULL, &status);     \
    checkResult(status, "CreateBuffer (output)");			\
    									\
    status = clGetKernelWorkGroupInfo(kernel, dId,			\
          CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t), &workgroup_size,0); \
    checkResult(status, "GetKernelWorkGroupInfo");			\
  }									\
									\
  void * svc(void *) {				                        \
    cl_int   status;							\
    cl_event events[2];							\
    size_t globalThreads[1];						\
    size_t localThreads[1];						\
    									\
    if (size  < workgroup_size) {					\
	localThreads[0]  = size;					\
	globalThreads[0] = size;					\
    } else {								\
	localThreads[0]=workgroup_size;					\
	globalThreads[0] = nextMultipleOfIf(size,workgroup_size);	\
    }									\
    									\
    status = clSetKernelArg(kernel, 0, sizeof(cl_mem),			\
			    (void *)&inputBuffer);			\
    checkResult(status, "setKernelArg 0");				\
    status = clSetKernelArg(kernel, 1, sizeof(cl_mem),			\
			    (void *)&outputBuffer);			\
    checkResult(status, "setKernelArg 1");				\
    status = clSetKernelArg(kernel, 2, sizeof(cl_uint),			\
			    (void *)&size);				\
    checkResult(status, "setKernelArg 2");				\
    									\
    status = clEnqueueWriteBuffer(cmd_queue,inputBuffer,CL_FALSE,0,	\
				  sizeof(cl_float ) * size,		\
				  &Vin[0],0,NULL,NULL);			\
    checkResult(status, "copying Vin to device input-buffer");		\
    									\
    status = clEnqueueNDRangeKernel(cmd_queue,kernel,1,NULL,		\
		    globalThreads,localThreads,0,NULL,&events[0]);      \
    									\
    status = clWaitForEvents(1, &events[0]);				\
    									\
    status = clEnqueueReadBuffer(cmd_queue,outputBuffer,CL_TRUE, 0,	\
		 size * sizeof(cl_float), &Vout[0],0,NULL,&events[1]);  \
    									\
    status = clWaitForEvents(1, &events[1]);				\
    status = clReleaseEvent(events[0]);					\
    status = clReleaseEvent(events[1]);					\
    return NULL;							\
  }									\
									\
  void svc_releaseOclObjects(){						\
    clReleaseKernel(kernel);						\
    clReleaseProgram(program);						\
    clReleaseCommandQueue(cmd_queue);					\
    clReleaseMemObject(inputBuffer);					\
    clReleaseMemObject(outputBuffer);					\
    clReleaseContext(context);						\
  }									\
									\
private:								\
   cl_float* Vin;							\
   cl_float* Vout;                                                      \
   size_t size;								\
   size_t workgroup_size;						\
   cl_context context;							\
   cl_program program;							\
   cl_command_queue cmd_queue;						\
   cl_mem inputBuffer;							\
   cl_mem outputBuffer;							\
   cl_kernel kernel;							\
   cl_int status;							\
}

/* TODO: pipe cleanup */
#define  GPUMAP(name, Vin, Vout, size)		                        \
    ff_pipeline pipe_##name;						\
    pipe_##name.add_stage(new gpu##name(Vin,Vout, size))

#define RUNGPUMAP(name)				                        \
    pipe_##name.run_and_wait_end()

#define GPUMAPWTIME(name)			                        \
    pipe_##name.ffwTime()

#endif    //OCL


/*++++++++++ node/dnode macroes +++++++++++ */

#define FFnode(fname,typein,typeout)                                    \
class _FF_node_##fname: public ff_node {				\
public:								        \
    void* svc(void* task) {						\
	typein* _ff_in = (typein*)task;					\
	typeout* res=NULL;						\
	fname(*_ff_in, res);						\
	delete _ff_in;							\
	if ((void*)res==GO_ON) return GO_ON;				\
	return res;							\
    }									\
}

#if !defined(FF_OCL)
#define FFDNODE_FNAME(fname, typein, typeout, COMM_IN, COMM_OUT)	\
class FFdnode_##fname: public ff_dnode<COMM_OUT> {			\
    typedef COMM_OUT::TransportImpl        transport_t;			\
protected:								\
    static void callback(void *e, void* ptr) {				\
	if (ptr == NULL) return;					\
	delete (typeout*)ptr;						\
    }									\
public:								        \
    FFdnode_##fname(unsigned nHosts, const std::string& name,		\
		    const std::string& address,				\
		    transport_t* const transp, const bool skip=false):	\
        nHosts(nHosts), name(name), address(address),                   \
	transp(transp), skip(skip) {}					\
									\
    int svc_init() {							\
	if (!skip) FFD_INIT_SENDER(COMM_OUT);				\
	return 0;							\
    }									\
    void * svc(void * task) {						\
        typein * tin = (typein*)task;					\
	typeout* res;							\
	fname(*tin,res);					        \
	tin->cleanup();							\
	delete (typein*)task;						\
	return res;							\
    }									\
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {	\
	typeout* t = (typeout*)ptr;					\
	int nnull=0;							\
	t->prepareOUT(v, nnull);					\
	for(int i=0;i<nnull;++i)					\
	    setCallbackArg(NULL);					\
	setCallbackArg(ptr);						\
    }									\
protected:								\
    const unsigned    nHosts;						\
    const std::string name;						\
    const std::string address;						\
    transport_t   * transp;						\
    const bool skip;							\
}

#define FFOCLMAP_FNAME(fname, typein, typeout)

#else
#define FFDNODE_FNAME(fname, typein, typeout, COMM_IN, COMM_OUT)   
#define FFOCLMAP_FNAME(fname, typein, typeout)				\
class oclTask: public baseTask {					\
protected:								\
 size_t s;								\
public:									\
 typedef double base_type;						\
									\
 oclTask():s(0) {}							\
 oclTask(base_type* t, size_t s):baseTask(t),s(s) {}			\
 /* FIX: SIZE is global !!!!!!!!!!! <------------------ */		\
 void   setTask(void* t) { if (t) { task=t; s= SIZE; } }		\
 size_t size() const     { return s;}					\
 size_t bytesize() const { return s*sizeof(base_type); }		\
 void*  newOutPtr()    { outPtr= new base_type[s]; return outPtr; }	\
 void   deleteOutPtr() { if (outPtr) delete [] outPtr; }		\
protected:								\
 base_type *outPtr;							\
};									\
									\
template<typename T>							\
class FFOclMap_##fname: public ff_mapOCL<T> {				\
public:									\
    typedef typename T::base_type base_type;				\
 FFOclMap_##fname(std::string code):ff_mapOCL<T>(code) {};		\
    void * svc(void * task) {						\
      typein * tin = (typein*)task;					\
      base_type *r = (base_type*)ff_mapOCL<T>::svc(tin);		\
      tin->cleanup();							\
      delete (typein*)task;						\
      return (new typeout(r));						\
  }									\
}
#endif


#define FFdnode(fname, typein, typeout, COMM_IN, COMM_OUT)              \
class GWin_##fname: public ff_dinout<COMM_IN, COMM_IN> {		\
    typedef COMM_IN::TransportImpl        transport_t;			\
protected:								\
    static void callback(void *e, void* ptr) {				\
	if (ptr == NULL) return;					\
	typein* _ptr = (typein*)ptr;					\
	_ptr->cleanup();						\
        delete _ptr;							\
    }									\
public:								        \
    GWin_##fname(unsigned nHosts, const std::string& name1,		\
		 const std::string& address1,				\
		 const std::string& name2, const std::string& address2,	\
		 transport_t* const transp):				\
    nHosts(nHosts), name1(name1),address1(address1),			\
	name2(name2),address2(address2),				\
	transp(transp) {}						\
									\
    int svc_init() {							\
        ff_dinout<COMM_IN,COMM_IN>::initOut(name2, address2, nHosts,	\
					    transp,transp->getProcId(), \
					    callback);			\
	ff_dinout<COMM_IN,COMM_IN>::initIn(name1, address1, 1, transp);	\
	return 0;							\
    }									\
    void * svc(void * task) { return task; }				\
									\
    void unmarshalling(svector<msg_t*>* const v[], const int vlen,	\
		       void *& task) {					\
	/* in general it is possible to avoid this copy..... */		\
	typein * t = new typein;					\
	t->unmarshall(v, vlen);						\
	task   = t;							\
    }									\
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {	\
	typein* t = (typein*)ptr;					\
	int nnull=0;							\
	t->prepareOUT(v, nnull);					\
	for(int i=0;i<nnull;++i)					\
	    setCallbackArg(NULL);					\
	setCallbackArg(ptr);						\
    }									\
protected:								\
    const unsigned    nHosts;						\
    const std::string name1;						\
    const std::string address1;						\
    const std::string name2;						\
    const std::string address2;						\
    transport_t   * transp;						\
};									\
class GWout_##fname: public ff_dinout<COMM_OUT, COMM_OUT> {		\
    typedef COMM_OUT::TransportImpl        transport_t;			\
protected:								\
    static void callback(void *e, void* ptr) {				\
	if (ptr == NULL) return;					\
	typeout* _ptr = (typeout*)ptr;					\
	_ptr->cleanup();						\
	delete _ptr;							\
    }									\
public:								        \
    GWout_##fname(unsigned nHosts, const std::string& name1,		\
		 const std::string& address1,				\
		 const std::string& name2, const std::string& address2,	\
		 transport_t* const transp):				\
    nHosts(nHosts), name1(name1),address1(address1),			\
	name2(name2),address2(address2),				\
	transp(transp) {}						\
									\
    int svc_init() {							\
	 ff_dinout<COMM_OUT,COMM_OUT>::initIn(name1, address1, nHosts,  \
                                              transp);			\
         ff_dinout<COMM_OUT,COMM_OUT>::initOut(name2, address2,1,	\
					       transp,			\
					       transp->getProcId(),	\
					       callback);		\
	 return 0;							\
    }									\
    void * svc(void * task) { return task; }				\
									\
    void unmarshalling(svector<msg_t*>* const v[], const int vlen,	\
		       void *& task) {					\
	/* in general it is possible to avoid this copy..... */		\
	typeout * t = new typeout;					\
	t->unmarshall(v, vlen);						\
	task   = t;							\
    }									\
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {	\
	typeout* t = (typeout*)ptr;					\
	int nnull=0;							\
	t->prepareOUT(v, nnull);					\
	for(int i=0;i<nnull;++i)					\
	    setCallbackArg(NULL);					\
	setCallbackArg(ptr);						\
    }									\
protected:								\
    const unsigned    nHosts;						\
    const std::string name1;						\
    const std::string address1;						\
    const std::string name2;						\
    const std::string address2;						\
    transport_t   * transp;						\
};									\
class INd_##fname: public ff_dnode<COMM_IN> {				\
    typedef COMM_IN::TransportImpl        transport_t;			\
public:								        \
    INd_##fname(unsigned nHosts, const std::string& name,		\
	const std::string& address, transport_t* const transp):		\
    nHosts(nHosts), name(name), address(address),transp(transp) {	\
	COUNT_TASKS_INIT;						\
    }									\
									\
    int svc_init() {							\
	FFD_INIT_RECEIVER(COMM_IN);					\
	return 0;							\
    }									\
    void * svc(void * task) { DO_COUNT_TASKS; return task; }		\
									\
    void unmarshalling(svector<msg_t*>* const v[], const int vlen,	\
		       void *& task) {					\
	/* in general it is possible to avoid this copy..... */		\
	typein * t = new typein;					\
	t->unmarshall(v, vlen);						\
	task   = t;							\
    }									\
    void svc_end() {							\
      PRINT_COUNT_TASKS;						\
    }									\
protected:								\
    const unsigned    nHosts;						\
    const std::string name;						\
    const std::string address;						\
    transport_t   * transp;						\
    COUNT_TASKS_DEFINE;							\
};									\
/* this is used only in case DNODE_FARM is defined */			\
class OUTd_##fname: public ff_dnode<COMM_OUT> {				\
    typedef COMM_OUT::TransportImpl        transport_t;			\
protected:								\
    static void callback(void *e, void* ptr) {				\
	if (ptr == NULL) return;					\
	delete (typeout*)ptr;						\
    }									\
public:								        \
    OUTd_##fname(unsigned nHosts, const std::string& name,		\
	const std::string& address, transport_t* const transp):		\
    nHosts(nHosts), name(name), address(address),transp(transp) {}	\
									\
    int svc_init() {							\
	FFD_INIT_SENDER(COMM_OUT);					\
	return 0;							\
    }									\
    void * svc(void * task) { return task; }				\
									\
    void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {	\
	typeout* t = (typeout*)ptr;					\
	int nnull=0;							\
	t->prepareOUT(v, nnull);					\
	for(int i=0;i<nnull;++i)					\
	    setCallbackArg(NULL);					\
	setCallbackArg(ptr);						\
    }									\
protected:								\
    const unsigned    nHosts;						\
    const std::string name;						\
    const std::string address;						\
    transport_t   * transp;						\
};									\
FFDNODE_FNAME(fname, typein, typeout, COMM_IN, COMM_OUT);		\
FFOCLMAP_FNAME(fname, typein, typeout)


#define FFnode_out(uname,typeout)                                       \
class _FF_node_##uname: public ff_node {				\
public:								        \
    void* svc(void*) {							\
	typeout* res;							\
	uname(res);							\
	return res;							\
    }									\
}

#define FFdnode_out(uname, typeout, COMM_OUT)	                        \
class FFdnode_##uname: public ff_dnode<COMM_OUT> {		        \
  typedef COMM_OUT::TransportImpl        transport_t;			\
protected:								\
  static void callback(void* e, void* ptr) {				\
      if (ptr==NULL) return;						\
      delete (typeout*)ptr;						\
  }									\
public:									\
    FFdnode_##uname(unsigned nHosts, const std::string& name,		\
          const std::string& address, transport_t* const transp):	\
    nHosts(nHosts),name(name),address(address),transp(transp) {}	\
                                                                        \
    int svc_init() {						        \
    	FFD_INIT_SENDER(COMM_OUT);					\
	return 0;							\
    }									\
    void * svc(void*) {							\
	typeout* res;							\
	uname(res);							\
	return res;							\
    }									\
									\
   void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {	\
       typeout* t = (typeout*)ptr;					\
       int nnull=0;							\
       t->prepareOUT(v,nnull);						\
       for(int i=0;i<nnull;++i)						\
	   setCallbackArg(NULL);					\
       setCallbackArg(ptr);						\
   }									\
protected:								\
   const unsigned nHosts;						\
   const std::string name;		 			        \
   const std::string address;						\
   transport_t   * transp;						\
}


#define FFnode_in(pname,typein)			                        \
class _FF_node_##pname: public ff_node {				\
public:								        \
    void* svc(void* task) {						\
	typein* res = (typein*)task;					\
	pname(res);							\
	delete res;							\
	return GO_ON;							\
    }									\
}

#define FFdnode_in(pname, typein, COMM_IN)	                        \
class FFdnode_##pname: public ff_dnode<COMM_IN> {		        \
     typedef COMM_IN::TransportImpl        transport_t;		        \
public:									\
     FFdnode_##pname(unsigned nHosts, const std::string& name,		\
         const std::string& address, transport_t* const transp):        \
     nHosts(nHosts), name(name), address(address),transp(transp) {}	\
                                                                        \
     int svc_init() {							\
	 FFD_INIT_RECEIVER(COMM_IN);					\
	 return 0;							\
     }									\
     void * svc(void * task) {						\
	 typein* m = (typein*)task;					\
	 pname(m);							\
	 delete m;							\
	 return GO_ON;							\
     }									\
									\
     void unmarshalling(svector<msg_t*>* const v[], const int vlen,	\
			void *& task) {					\
	 typein* t = new typein;					\
	 t->unmarshall(v, vlen);					\
	 t->cleanup();							\
	 task = t;							\
     }									\
protected:								\
     const unsigned    nHosts;						\
     const std::string name;						\
     const std::string address;						\
     transport_t   * transp;						\
}

/*++++++++++ init/end macroes +++++++++++ */

#define FFd_init(hostid)			                        \
    zmqTransport FFd_transport(hostid);					\
    if (FFd_transport.initTransport()<0) abort()

#define FFd_end()			                                \
    FFd_transport.closeTransport()

#define FFdnode_out_run(nodename,nHosts,chname,address,transport)	\
    FFdnode_##nodename __##nodename(nHosts,chname,address,transport);	\
    __##nodename.skipfirstpop(true);					\
    __##nodename.run();							\
    __##nodename.wait()

#define FFdnode_in_run(nodename,nHosts,chname,address,transport)	\
    FFdnode_##nodename __##nodename(nHosts,chname,address,transport);	\
    __##nodename.run();							\
    __##nodename.wait()

#define FFdgw_in_run(nodename,nHosts,chname1,address1,			\
		     chname2,address2, transport)			\
    GWin_##nodename __##nodename(nHosts,chname1,address1,		\
				 chname2,address2, transport);		\
    __##nodename.run();							\
    __##nodename.wait()

#define FFdgw_out_run(nodename,nHosts,chname1,address1,			\
		      chname2,address2,transport)			\
    GWout_##nodename __##nodename(nHosts,chname1,address1,		\
				  chname2,address2,transport);		\
    __##nodename.run();							\
    __##nodename.wait()

#define FFdnode_run(nodename, chnameIN, addressIN,			\
                    chnameOUT,addressOUT, transport)			\
    INd_##nodename   __ind_##nodename(1,chnameIN,addressIN,transport);	\
    FFdnode_##nodename __##nodename(1,chnameOUT,addressOUT,transport);	\
    ff_pipeline __pipe;							\
    __pipe.add_stage(&__ind_##nodename);				\
    __pipe.add_stage(&__##nodename);					\
    __pipe.run_and_wait_end()

/* TODO: farm cleanup */
#define FFdnode_farmrun(pardegree, nodename, chnameIN, addressIN,	\
			chnameOUT,addressOUT, transport)		\
    ff_farm<>   __ffdfarm;						\
    std::vector<ff_node*> w_ffdfarm;					\
    for(int i=0;i<pardegree;++i)					\
	w_ffdfarm.push_back(new FFdnode_##nodename(1,"",addressIN,	\
						   transport, true));	\
    __ffdfarm.add_emitter(new INd_##nodename(1,chnameIN,addressIN,	\
					     transport));		\
    __ffdfarm.add_collector(new OUTd_##nodename(1,chnameOUT,		\
						addressOUT,transport));	\
    __ffdfarm.add_workers(w_ffdfarm);					\
    /* scheduling policy */						\
    __ffdfarm.set_scheduling_ondemand();				\
    __ffdfarm.run_and_wait_end()

/* TODO: farm cleanup */
#define FFOclMap_farmrun(pardegree, nodename, chnameIN, addressIN,	\
			 chnameOUT,addressOUT, transport)		\
    ff_farm<>   __ffdfarm;						\
    std::vector<ff_node*> w_ffdfarm;					\
    for(int i=0;i<pardegree;++i)					\
	w_ffdfarm.push_back(new FFOclMap_##nodename<oclTask>(nodename));	\
    __ffdfarm.add_emitter(new INd_##nodename(1,chnameIN,addressIN,	\
					     transport));		\
    __ffdfarm.add_collector(new OUTd_##nodename(1,chnameOUT,		\
						addressOUT,transport));	\
    __ffdfarm.add_workers(w_ffdfarm);					\
    /* scheduling policy */						\
    __ffdfarm.set_scheduling_ondemand();				\
    __ffdfarm.run_and_wait_end()


/* ------------------------ high-level patterns --------------------------- */

#define PIPE3DEF(first,middle,last, typein, typeout)			\
    FFdnode_out(first, typein, zmqOnDemand);				\
    FFdnode(middle, typein,typeout,zmqOnDemand, zmqFromAny);		\
    FFdnode_in(last, typeout, zmqFromAny)



// TODO: pipe cleanup
#define PIPE3(pipename, first,middle,last)		\
    ff_pipeline pipename;				\
    pipename.add_stage(new _FF_node_##first);		\
    pipename.add_stage(new _FF_node_##middle);		\
    pipename.add_stage(new _FF_node_##last)

// just defines a pipeline 
#define PIPE(pipename)				        \
    ff_pipeline pipename

#define PIPEADDPTR(pipename, stage)		        \
    pipename.add_stage(&_FF_node_##stage)

#define PIPEADD(pipename, stage)		        \
    pipename.add_stage(new _FF_node_##stage)

#define RUNPIPE(pipename)			        \
    pipename.run_and_wait_end()

// TODO: farm cleanup 
#define FARM(farmname, worker, nworkers)		\
    ff_farm<> _FF_node_##farmname;			\
    std::vector<ff_node*> w_##farmname;			\
    for(int i=0;i<nworkers;++i)                         \
      w_##farmname.push_back(new _FF_node_##worker);	\
    _FF_node_##farmname.add_workers(w_##farmname);	\
    _FF_node_##farmname.add_collector(NULL);


// TODO: farm cleanup 
#define FARMA(farmname, worker, nworkers)		\
    ff_farm<> _FF_node_##farmname(true);		\
    std::vector<ff_node*> w_##farmname;			\
    for(int i=0;i<nworkers;++i)                         \
	w_##farmname.push_back(new _FF_node_##worker);	\
    _FF_node_##farmname.add_workers(w_##farmname);	\
    _FF_node_##farmname.add_collector(NULL);

// TODO: farm cleanup 
#define FARMA2(farmname, worker, nworkers)		\
    ff_farm<> _FF_node_##farmname(true);		\
    std::vector<ff_node*> w_##farmname;			\
    for(int i=0;i<nworkers;++i)                         \
	w_##farmname.push_back(new _FF_node_##worker);	\
    _FF_node_##farmname.add_workers(w_##farmname);	\


#define MAPDEF(mapname,func, basictype)					\
    void* mapdef_##mapname(basePartitioner*const P,int tid) {		\
       LinearPartitioner<basictype>* const partitioner=			\
	(LinearPartitioner<basictype>* const)P;				\
       LinearPartitioner<basictype>::partition_t Partition;		\
       partitioner->getPartition(tid, Partition);			\
       basictype* p = (basictype*)(Partition.getData());		\
       size_t l = Partition.getLength();				\
       func(p,l);							\
       return p;							\
    }

#define MAP(mapname, basictype, V,size,nworkers)		\
    LinearPartitioner<basictype> P(size,nworkers);		\
    ff_map _map_##mapname(mapdef_##mapname,&P,V)

#define MAPDEF1(mapname,func, basictype)				\
    void* mapdef_##mapname(basePartitioner*const P,int tid) {		\
       LinearPartitioner<basictype>* const partitioner=			\
	(LinearPartitioner<basictype>* const)P;				\
       LinearPartitioner<basictype>::partition_t Partition;		\
       partitioner->getPartition(tid, Partition);			\
       basictype* p = (basictype*)(Partition.getData());		\
       size_t l = Partition.getLength();				\
       for(size_t i=0;i<l;++i)						\
	   func(p[i]);							\
       return p;							\
    }



#define RUNFARMA(farmname)			\
    _FF_node_##farmname.run_then_freeze()

#define RUNMAP(mapname)				\
    _map_##mapname.run_and_wait_end()

#define FARMAOFFLOAD(farmname, task)		\
    _FF_node_##farmname.offload(task);

#define FARMAGETRESULTNB(farmname, result)	\
    _FF_node_##farmname.load_result_nb((void**)result)

#define FARMAGETRESULT(farmname, result)	\
    _FF_node_##farmname.load_result((void**)result)

#define FARMAWAIT(farmname)			\
    _FF_node_##farmname.wait()

#define FARMAEND(farmname)			\
    _FF_node_##farmname.offload(EOS)

#define FARMAWTIME(farmname)			\
    _FF_node_##farmname.ffwTime()


#if (FFDISTRIBUTED != 1)
/* non distributed version */
/* TODO: pipe and farm cleanup */
#define FFd_PIPE3(hostid, first, middle, last,address1,address2,par1, ...)           \
    ff_farm<>  __ffdfarm3;						             \
    std::vector<ff_node*> w_ffdfarm3;					             \
    for(int i=0;i<par1;++i)						             \
	w_ffdfarm3.push_back(new FFdnode_##middle(1,"A",address2,&FFd_transport));   \
    __ffdfarm3.add_workers(w_ffdfarm3);					             \
    __ffdfarm3.add_collector(NULL);					             \
    /* scheduling policy */						             \
    __ffdfarm3.set_scheduling_ondemand();				             \
    ff_pipeline __ffdpipe3;						             \
    __ffdpipe3.add_stage(new FFdnode_##first(1,"A",address1,&FFd_transport));        \
    __ffdpipe3.add_stage(&__ffdfarm3);					             \
    __ffdpipe3.add_stage(new FFdnode_##last(1,"A",address1,&FFd_transport));         \
    __ffdpipe3.run_and_wait_end()

#else /* FFDISTRIBUTED */
/* distributed version */
#if !defined(DNODE_FARM)
#define FFd_PIPE3(hostid, first, middle, last,address1,address2,par1, ...) \
    switch(hostid) {							   \
    case -1: {								   \
	FFdnode_out_run(first, par1,"A",address1,&FFd_transport);	   \
    } break;								   \
    case -2: {								   \
	FFdnode_in_run(last, par1, "B",address2,&FFd_transport);	   \
    } break;								   \
    case -3: {								   \
        FFdgw_in_run(middle,par1,					   \
                     "A",address1,"A",address2,&FFd_transport);		   \
    } break;								   \
    case -4: {								   \
	FFdgw_out_run(middle,par1,					   \
		      "B",address1,"B",address2,&FFd_transport);	   \
    } break;								   \
    default: {								   \
	FFdnode_run(middle, "A",address1,"B",address2,&FFd_transport);	   \
    }									   \
    }
#else  /* DNODE_FARM */
#define FFd_PIPE3(hostid, first, middle, last,address1,address2,par1,par2) \
    switch(hostid) {							   \
    case -1: {								   \
	FFdnode_out_run(first, par1,"A",address1,&FFd_transport);	   \
    } break;								   \
    case -2: {								   \
	FFdnode_in_run(last, par1, "B",address2,&FFd_transport);	   \
    } break;								   \
    case -3: {								   \
	FFdgw_in_run(middle,par1,					   \
		     "A",address1,"A",address2,&FFd_transport);		   \
    } break;								   \
    case -4: {								   \
	FFdgw_out_run(middle,par1,					   \
		      "B",address1,"B",address2,&FFd_transport);	   \
    } break;								   \
    default: {								   \
	FFdnode_farmrun(par2, middle,"A",address1,"B",address2,		   \
			&FFd_transport);				   \
    }									   \
    }

#define FFd_PIPE3MAP(hostid, first, middle, last,address1,address2,par1,par2) \
    switch(hostid) {							   \
    case -1: {								   \
	FFdnode_out_run(first, par1,"A",address1,&FFd_transport);	   \
    } break;								   \
    case -2: {								   \
	FFdnode_in_run(last, par1, "B",address2,&FFd_transport);	   \
    } break;								   \
    case -3: {								   \
	FFdgw_in_run(middle,par1,					   \
		     "A",address1,"A",address2,&FFd_transport);		   \
    } break;								   \
    case -4: {								   \
	FFdgw_out_run(middle,par1,					   \
		      "B",address1,"B",address2,&FFd_transport);	   \
    } break;								   \
    default: {								   \
	FFOclMap_farmrun(par2, middle,"A",address1,"B",address2,	   \
			 &FFd_transport);				   \
    }									   \
    }

#endif /* DNODE_FARM */
#endif /* FFDISTRIBUTED */


#endif /* __FF_MACROES_HPP_ */
