#pragma once

#define CL_HPP_CL_1_2_DEFAULT_BUILD
#define CL_HPP_TARGET_OPENCL_VERSION 120
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_ENABLE_PROGRAM_CONSTRUCTION_FROM_ARRAY_COMPATIBILITY 1

#include <CL/cl2.hpp>
#include <iostream>
#include <fstream>
#include <iostream>
#include <string>
// pre unique_ptr
#include <memory>
#include "task.hpp"

// per fastflow
#include <ff/ff.hpp>

#define OCL_CHECK(error, call)						\
  call;									\
  if (error != CL_SUCCESS) {						\
    printf("%s:%d Error calling " #call ", error code is: %d\n", __FILE__, __LINE__, error); \
    exit(EXIT_FAILURE);							\
  }


//  questa classe serve per trovare un device compatibile e per permetterne l'utilizzo

class FpgaDevice {

private: 
  std::vector<cl::Device> devices;
  cl::Device device;
  cl_int err;
  cl::Context context;
  cl::CommandQueue q;
  cl::Kernel krnl;
  cl::Program program;
  std::vector<cl::Platform> platforms;
  bool found_device = false;
  // aggiunti per parametri
  // cl::Buffer *buffer_a;
  // cl::Buffer *buffer_b;
  // cl::Buffer *buffer_result;
  // int *ptr_a;
  // int *ptr_b;
  // int *ptr_result;

  std::vector<std::pair<cl::Buffer *, size_t>> in_buffers;  // vecotr of (dev_ptr, nax_size_in_bytes) for IN device buffers
  std::vector<std::pair<cl::Buffer *, size_t>> out_buffers; // vecotr of (dev_ptr, nax_size_in_bytes) for OUT device buffers

public: 

  // costruttore: controlla che ci sia un device accessibile 
  FpgaDevice() {
    // trova la piattaforma
    cl::Platform::get(&platforms);
    for(size_t i = 0; (i < platforms.size() ) & (found_device == false) ;i++){
      cl::Platform platform = platforms[i];
      std::string platformName = platform.getInfo<CL_PLATFORM_NAME>();
      if ( platformName == "Xilinx"){
	devices.clear();
	// trova i device
	platform.getDevices(CL_DEVICE_TYPE_ACCELERATOR, &devices);
	if (devices.size()){
	  device = devices[0];
	  found_device = true;
	  break;
	}
      }
    }
    std::cerr << "MDLOG: Device found " << std::endl; 
    context = cl::Context(device, NULL, NULL, NULL, &err);
    if(err != CL_SUCCESS) {
      found_device = false; 
      return;
    }
    std::cerr << "MDLOG: got context " << std::endl; 
    // crea la command queue
    q = cl::CommandQueue(context, device, CL_QUEUE_PROFILING_ENABLE, &err);
    if(err != CL_SUCCESS) {
      found_device = false; 
      return;
    }
    std::cerr << "MDLOG: got queue " << std::endl; 

    // se esci da qui è stato trovato il device e la command queue è a posto 
    return; 
  }

  bool DeviceExists() {
    return(found_device);
  }


  bool CreateKernel(std::string xclbinFilename, const std::string kernelName) {
    // open file
    FILE* fp;
    if ((fp = fopen(xclbinFilename.c_str(), "r")) == nullptr) {
      printf("ERROR: %s xclbin not available please build\n", xclbinFilename.c_str());
      return(false);
    }
    // Load xclbin
    std::ifstream bin_file(xclbinFilename, std::ifstream::binary);
    bin_file.seekg (0, bin_file.end);
    unsigned nb = bin_file.tellg();
    bin_file.seekg (0, bin_file.beg);
    char *buf = new char [nb];
    bin_file.read(buf, nb);
   
    // Creating Program from Binary File
    cl::Program::Binaries bins;
    bins.push_back({buf,nb});
    devices.resize(1);
    program = cl::Program(context, devices, bins, NULL, &err);
    if(err != CL_SUCCESS) {
      std::cout << "Error loading program code from xclbin" << std::endl;
      return(false);
    }
   
    krnl = cl::Kernel(program,kernelName.c_str(), &err);
    if(err != CL_SUCCESS) {
      std::cout << "Error creating kernel from xclbin" << std::endl;
      return(false);
    }

    return(true);
  }

  // NOTE: Buffers both IN and OUT are not allowed!
  bool setupBuffers(FTask buff_description)
  {
    int narg = 0;

    for (auto & p : buff_description.in) {
      auto buff = new cl::Buffer(context, CL_MEM_READ_ONLY, p.second, nullptr, &err);
      if (err != CL_SUCCESS) return false;
      in_buffers.push_back({buff, p.second});

      krnl.setArg(narg++, *buff);
    }

    for (auto & p : buff_description.out) {
      auto buff = new cl::Buffer(context, CL_MEM_WRITE_ONLY, p.second, nullptr, &err);
      if (err != CL_SUCCESS) return false;
      out_buffers.push_back({buff, p.second});

      krnl.setArg(narg++, *buff);
    }

    return true;
  }


  bool computeTask(FTask * task) {
    int i = 0;
    for (auto & p : task->in) {
      err = q.enqueueWriteBuffer(*(in_buffers[i++].first), CL_TRUE, 0, p.second, p.first);
      if (err != CL_SUCCESS) return false;
    }

    // TODO: check if this code works for scalars
    i += task->out.size();
    for (auto & p : task->scalars) {
      krnl.setArg(i++, *((int*)p.first));
    }

    err = q.enqueueTask(krnl);
    if (err != CL_SUCCESS) return false;


    i = 0;
    for (auto & p : task->out) {
      err = q.enqueueReadBuffer(*(out_buffers[i++].first), CL_TRUE, 0, p.second, p.first);
      if (err != CL_SUCCESS) return false;
    }

    return true;
  }

  bool waitTask() {
    err = q.finish();
    if(err != CL_SUCCESS) 
      return(false);
    return(true); 
  }

  bool finalize() {

    // check allocated buffers to be destroyed
    err = q.finish();
    if(err != CL_SUCCESS) 
      return(false);
    return(true);
  }

  void cleanup() {
    for (auto p : in_buffers) {
      delete p.first;
    }

    for (auto p : out_buffers) {
      delete p.first;
    }
  }
};
