#ifndef _SPD_PIPE_OCL_
#define _SPD_PIPE_OCL_

//OpenCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

//FF
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/node.hpp>
#include <ff/oclnode.hpp>
//#include <ff/ocl/mem_man.h>
using namespace ff;

#include <fstream>
#include <sstream>
using namespace std;

#include "utils.hpp"
#include "noise_detection.hpp"

#define EPS_STOP 1e-04f



/*
  DETECT STAGE
 */
class Detect: public ff_node {
public:
  Detect(unsigned int w_max): w_max(w_max) {}

  void * svc(void * task) {
#ifdef TIME
    unsigned long time_ = get_usec_from(0);
#endif
    task_frame *t = (task_frame *)task;
    vector<unsigned int> noisy_;
    t->noisymap = new cl_int[t->height * t->width];
    detect(noisy_, t->noisymap, t->in, t->height, t->width, w_max);
    t->n_noisy = (cl_uint)noisy_.size();
    t->noisy = new cl_uint[t->n_noisy];
    for(unsigned int i=0; i<t->n_noisy; ++i)
      (t->noisy)[i] = noisy_[i];
#ifdef TIME
    t->svc_time_detect = get_usec_from(time_);
#endif
    return task;
  }

private:
  unsigned int w_max;

  void detect(
	      vector<unsigned int> &noisy,
	      cl_int *noisymap,
	      cl_uchar *im,
	      unsigned int height,
	      unsigned int width,
	      unsigned int w_max
	      )
  {
    //linearized control window
    cl_uchar c_array[MAX_WINDOW_SIZE * MAX_WINDOW_SIZE];
    unsigned int first_row = 0, last_row = height-1;
    for(unsigned int r=first_row; r<=last_row; ++r) {
      for(unsigned int c=0; c<width; c++) {
	unsigned int idx = r * width + c;
	cl_uchar center = im[idx];
	if(is_noisy<cl_uchar>(center, r, c, w_max, c_array, im, width, height)) {
	  noisy.push_back(idx);
	  noisymap[idx] = center;
	}
	else
	  noisymap[idx] = -1;
      }
    }
  }
};





/*
  DENOISE (farm-)WORKER
 */
class DenoiseW: public ff_oclNode {
public:
  DenoiseW(float alpha, float beta)
    : alpha(alpha), beta(beta) {}

  void svc_SetUpOclObjects(cl_device_id dId){
    cerr << "device: " << dId << endl;
    cl_int res;

    //create context
    context = clCreateContext (NULL, 1, &dId, NULL, NULL, &res);
    check(res, "creating context");

    //create a command-queue
    command_queue = clCreateCommandQueue (context, dId, 0, &res);
    check(res, "creating command-queue");

    //create a CL program using the kernel source
    std::string kernelPath = KERNEL_PATH; //"../../../../examples/spd-denoiser/opencl/kernel_ocl/fuy.cl";  // ../kernel_ocl/fuy.cl";
    std::ifstream t(kernelPath.c_str());
    std::stringstream buffer;
    buffer << t.rdbuf();
    std::string source_string = buffer.str() + '\0';
    const char * source =  source_string.c_str();
    size_t sourcelen = strlen(source);
    cl_program program = clCreateProgramWithSource(context, 1, &source, &sourcelen, &res);
    check(res, "creating program with source");
    string clBuildOption = "-cl-fast-relaxed-math";
    res = clBuildProgram(program, 1, &dId, clBuildOption.c_str(), NULL, NULL);
    check(res, "building program");

    // size_t len;
    // char *buff;
    // clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);
    // buff = new char[len];
    // clGetProgramBuildInfo(program, dId, CL_PROGRAM_BUILD_LOG, len, buff, NULL);
    // cerr << buff << endl;

    //create fuy kernel
    kernel = clCreateKernel(program, "fuy", &res);
    check(res, "creating kernel: fuy");

    //create init kernel
    kernel_init = clCreateKernel(program, "init", &res);
    check(res, "creating kernel: init");
    
    /* Check group size against kernelWorkGroupSize */
    res = clGetKernelWorkGroupInfo(kernel, dId,CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t), &workgroup_size,0);
  }

  bool device_rules(cl_device_id dId){ 
    size_t len;
    cl_device_type d_type;
    clGetDeviceInfo(dId, CL_DEVICE_VENDOR, 0, NULL, &len);
    char* vendor = new char[len];
    clGetDeviceInfo(dId, CL_DEVICE_VENDOR, len, vendor, NULL);
    printf(" the device vendor is %s\n", vendor);
    //clGetDeviceInfo(dId, CL_DEVICE_TYPE, sizeof(cl_device_type), &(d_type), NULL);

    //workgroups size
    //clGetDeviceInfo(dId, CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &workgroup_size, NULL);

    return true;
  }
   
  void * svc(void * task) {
#ifdef TIME
    unsigned long time_ = get_usec_from(0);
#endif
    task_frame *t = (task_frame *)task;
    cl_uchar *clNoisy = t->in;
    unsigned int height = t->height;
    unsigned int width = t->width;
    t->out = new cl_uchar[height * width];
    cl_uchar *clRestored = t->out;
    cl_uint *noisy = t->noisy;
    cl_int *noisymap = t->noisymap;
    unsigned int n_noisy = t->n_noisy;
    bool fixed_cycles = t->fixed_cycles;
    int max_cycles = t->max_cycles;
    cl_int res;

    if(n_noisy > 0) {

    size_t n_workgroups = (n_noisy + (workgroup_size - 1)) / workgroup_size; //round-up
    size_t work_items = n_workgroups * workgroup_size;
    //cerr << "workgroup size: " << workgroup_size
    // 	 << ", n. workgroups: " << n_workgroups
    //  	 << ", work items: " << work_items << " (" << n_noisy << " noisy)" << endl;

    // unsigned long checksum = 0;
    // for(unsigned int i=0; i<height; ++i)
    //   for(unsigned int j=0; j<width; ++j)
    // 	checksum += clNoisy[i * width + j];
    // cerr << "starting checksum = " << checksum << endl;

    //allocate device memory
    cl_mem inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_uchar) * width * height, NULL, &res);
    cl_mem noisyBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_uint) * n_noisy, NULL, &res);
    cl_mem noisymapBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_int) * width * height, NULL, &res);
    cl_mem diffBuffer = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(cl_uchar) * n_noisy, NULL, &res);
    cl_mem residualBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_float) * n_workgroups, NULL, &res); //groupwise residuals
    cl_mem outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uchar) * width * height, NULL, &res);
    cl_mem flagBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int), NULL, &res);

    //set fuy-kernel args
    int argid = 0;
    res = clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&outputBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&inputBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&noisyBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&noisymapBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&diffBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&residualBuffer);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_float) * workgroup_size, NULL); //local residuals
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_uint), (void *)&workgroup_size);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_uint), (void *)&n_noisy);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_uint), (void *)&height);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_uint), (void *)&width);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_float), (void *)&alpha);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_float), (void *)&beta);
    res &= clSetKernelArg(kernel, argid++, sizeof(cl_mem), (void *)&flagBuffer);
    check(res, "setting kernel argumemts");

    //set init-kernel args
    argid = 0;
    res = clSetKernelArg(kernel_init, argid++, sizeof(cl_mem), (void *)&diffBuffer);
    res &= clSetKernelArg(kernel_init, argid++, sizeof(cl_mem), (void *)&residualBuffer);
    res &= clSetKernelArg(kernel_init, argid++, sizeof(cl_uint), (void *)&n_noisy);
    res &= clSetKernelArg(kernel_init, argid++, sizeof(cl_mem), (void *)&flagBuffer);
    check(res, "setting kernel argumemts");

    //*** RUNTIME
    //copy to device buffer
    cl_event mem_event;
    res = clEnqueueWriteBuffer(command_queue, inputBuffer, CL_TRUE, 0, sizeof(cl_uchar) * height * width, clNoisy, 0, NULL, &mem_event);
    check(res, "copying image to device input-buffer");
    clWaitForEvents(1, &mem_event);
    res = clEnqueueWriteBuffer(command_queue, outputBuffer, CL_TRUE, 0, sizeof(cl_uchar) * height * width, clNoisy, 0, NULL, &mem_event);
    check(res, "copying image to device output-buffer");
    clWaitForEvents(1, &mem_event);
    res = clEnqueueWriteBuffer(command_queue, noisyBuffer, CL_TRUE, 0, sizeof(cl_uint) * n_noisy, noisy, 0, NULL, &mem_event);
    check(res, "copying noisy to device buffer");
    clWaitForEvents(1, &mem_event);
    res = clEnqueueWriteBuffer(command_queue, noisymapBuffer, CL_TRUE, 0, sizeof(cl_int) * height * width, noisymap, 0, NULL, &mem_event);
    check(res, "copying noisymap to device buffer");
    clWaitForEvents(1, &mem_event);

    //init convergence structures
    cl_event end_init;
    res = clEnqueueNDRangeKernel(command_queue, kernel_init, 1, NULL, &work_items, &workgroup_size, 0, NULL, &end_init);
    check(res, "executing kernel: init convergence");
    res = clWaitForEvents(1, &end_init);


    unsigned int cycle = 0;
    bool fix = false;
    cl_float delta, new_residual;
    cl_float residual = 0.0f;
    cl_float *residuals = new cl_float[n_workgroups];
    while(!fix) {
      ++cycle;
      //restoration pass (kernel)
      cl_event end_pass;
      res = clEnqueueNDRangeKernel(command_queue, kernel, 1, NULL, &work_items, &workgroup_size, 0, NULL, &end_pass);
      check(res, "executing kernel: fuy");
      res = clWaitForEvents(1, &end_pass);

      //read back from device buffers
      res = clEnqueueReadBuffer (command_queue, residualBuffer, CL_TRUE, 0, sizeof(cl_float) * n_workgroups, residuals, 0, NULL, NULL);
      check(res, "copying back residuals from device buffer");

      //check convergence
      cl_float new_residual, sum = 0.0f;
      for(int i=0; i<n_workgroups; ++i) {
	sum += residuals[i];
      }
      new_residual = sum / n_noisy;

      delta = _ABS(new_residual - residual);
      
      // //read back the restored bitmap
      // res = clEnqueueReadBuffer (command_queue, outputBuffer, CL_TRUE, 0, sizeof(cl_uchar) * height * width, clRestored, 0, NULL, NULL);
      // check(res, "copying back restored from device buffer");
      // long chk = 0.0f;
      // for(unsigned int i=0; i<height; ++i)
      // 	for(unsigned int j=0; j<width; ++j)
      // 	  chk += clRestored[i*width + j];
      // cerr << "cycle " << cycle << " checksum = " << chk << endl;

      fix = cycle == max_cycles;
      if(!fixed_cycles)
	fix |= delta < EPS_STOP;

      if(!fix) {
	residual = new_residual;
	//copy output into input
	cl_event end_copy;
	res = clEnqueueCopyBuffer (command_queue, outputBuffer, inputBuffer, 0, 0, sizeof(cl_uchar) * height * width, 0, NULL, &end_copy);
	res = clWaitForEvents(1, &end_copy);
      }
    }

    //read back the restored bitmap
    res = clEnqueueReadBuffer (command_queue, outputBuffer, CL_TRUE, 0, sizeof(cl_uchar) * height * width, clRestored, 0, NULL, NULL);
    check(res, "copying back restored from device buffer");

    //clean-up
    delete[] residuals;
    clReleaseMemObject(inputBuffer);
    clReleaseMemObject(noisyBuffer);
    clReleaseMemObject(noisymapBuffer);
    clReleaseMemObject(diffBuffer);
    clReleaseMemObject(residualBuffer);
    clReleaseMemObject(outputBuffer);

    t->cycles = cycle;
    }
#ifdef TIME
    t->svc_time_denoise = get_usec_from(time_);
#endif
    return task;
  }
  
  void svc_releaseOclObjects() {
    clReleaseKernel(kernel);
    clReleaseKernel(kernel_init);
    //clReleaseProgram(program); crash - TODO: why?
    clReleaseCommandQueue(command_queue);
    clReleaseContext(context);
  }

private:
  size_t workgroup_size;
  cl_context context;
  cl_program program;
  cl_command_queue command_queue;
  cl_kernel kernel, kernel_init;
  cl_float alpha, beta;
};





/*
  DENOISE (farm-)COLLECTOR
 */
#include <queue>

class task_frame_comparison {
public:
  bool operator() (task_frame * &lhs, task_frame * &rhs) {
    // lhs < rhs (i. e. lhs has lower priority)
    // if rhs is an earlier frame
    return lhs->frame_id > rhs->frame_id;
  }
};

class DenoiseC : public ff_node {
public:
  DenoiseC() {
    id_expected = 1;
  }

  void * svc(void * task) {
    task_frame *t = (task_frame *)task;
    //cerr << "[expected #"<<id_expected<<"] received task #" << t->frame_id << endl;
    buffer.push(t);
    while(!buffer.empty()) {
      t = buffer.top();
      //cerr << "trying: read " << t->frame_id << " vs " << id_expected << endl;
      if(t->frame_id == id_expected) {
	//cerr << "sending task #" << t->frame_id << endl;
	ff_send_out((void *)t);
	buffer.pop();
	++id_expected;
      }
      else
	break;
    }
    return GO_ON;
  }

private:
  priority_queue<task_frame *, vector<task_frame *>, task_frame_comparison> buffer;
  unsigned int id_expected;
};





typedef struct pipe_components {
  unsigned int GPU_COUNT;
  ff_node *detect_stage, *denoise_stage;
  DenoiseC *denoise_collector;
  vector<ff_node *> denoise_workers;
} pipe_components_t;

void setup_pipe(ff_pipeline &pipe, pipe_components_t &comp, unsigned int w_max, float alfa, float beta, unsigned int GPU_COUNT) {
  comp.GPU_COUNT = GPU_COUNT;
  comp.detect_stage = new Detect(w_max);
  if(GPU_COUNT > 1) { //to be changed
    comp.denoise_stage = new ff_farm<>();
    for(unsigned int i=0; i<GPU_COUNT; ++i)
      comp.denoise_workers.push_back(new DenoiseW(alfa, beta));
    ((ff_farm<> *)(comp.denoise_stage))->add_workers(comp.denoise_workers);
    comp.denoise_collector = new DenoiseC();
    ((ff_farm<> *)(comp.denoise_stage))->add_collector(comp.denoise_collector);
  }
  else
    comp.denoise_stage = new DenoiseW(alfa, beta);
  //add stages
  pipe.add_stage(comp.detect_stage);
  pipe.add_stage(comp.denoise_stage);
}

void clean_pipe(pipe_components_t &comp) {
  delete (Detect *)(comp.detect_stage);
  if(comp.GPU_COUNT > 1) {
    for(unsigned int i=0; i<comp.GPU_COUNT; ++i)
      delete (DenoiseW *)(comp.denoise_workers[i]);
    delete (DenoiseC *)(comp.denoise_collector);
    delete (ff_farm<> *)(comp.denoise_stage);
  }
  else
    delete (DenoiseW *)(comp.denoise_stage);
}
#endif
