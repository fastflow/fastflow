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
 Mehdi Goli: m.goli@rgu.ac.uk */

/*
 * Very basic test for the heterogeneous FastFlow pipeline (actually a 3-stage torus).
 *
 */

#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/oclnode.hpp>

#include <iostream>
#include <sstream>
#include <string>
#include <stdio.h>

#include <cstring>
#include <cstdlib>
#include <unistd.h>
#include <vector>
#include <string>
#include <fstream>
#include <malloc.h>
#include <sys/time.h>
//#include "mpi.h"

/* 
 * THRESHOLD is the static estimation of the maximum number of ff_node which can run on GPU. 
 * By default the value is set to 1 ff_node per GPU.
 */
//#define THRESHOLD 4 


cl_int width =4096;
cl_int height=4096;
cl_uint maskWidth=8;
cl_uint maskHeight=8;
cl_int worker_width=8;
cl_int worker_height=8;
cl_uint2 inputDimensions = {width    , height};
cl_uint2 maskDimensions  = {maskWidth, maskHeight};
int quiet=1;
const int nworkers=8;
const int gnworkers=6;
const int max_strlen=10;
double get_current_time();
using namespace std;
using namespace ff;

struct task_t{
  task_t(){
    inpt= new cl_uint[ width * height];
    outpt= new cl_uint[width*height];
    msk=new cl_float[maskWidth * maskHeight];
  }
  cl_uint* inpt;
  cl_uint* outpt;
  cl_float* msk;
    //  long int id;
};

//struct task_t* t;
struct task_worker{
    task_worker(int i,int j, task_t * t):i(i),j(j),t(t){}
    int i;
    int j;
    task_t * t;
  };

template<typename T> 
void printArray(std::string header, const T * data,  const int width,const int height)
{
    std::cout<<"\n"<<header<<"\n";
    for(int i = 0; i < height; i++)
    {
        for(int j = 0; j < width; j++)
        {
            std::cout<<data[i*width+j]<<" ";
        }
        std::cout<<"\n";
    }
    std::cout<<"\n";
}


int fillRandom(cl_uint * arrayPtr,  const int width,const int height,const cl_uint rangeMin,const cl_uint rangeMax,unsigned int seed)
{
    if(!arrayPtr)
    {
        printf("Cannot fill array. NULL pointer.");
        return 0;
    }

    if(!seed)
        seed = (unsigned int)time(NULL);

    srand(seed);
    double range = double(rangeMax - rangeMin) + 1.0; 

    /* random initialisation of input */
    for(int i = 0; i < height; i++)
        for(int j = 0; j < width; j++)
        {
            int index = i*width + j;
            arrayPtr[index] = rangeMin + (cl_uint)(range*rand()/(RAND_MAX + 1.0)); 
        }

    return 1;
}


template<typename T>
T roundToPowerOf2(T val)
{
    int bytes = sizeof(T);

    val--;
    for(int i = 0; i < bytes; i++)
        val |= val >> (1<<i);  
    val++;

    return val;
}

template<typename T>
int isPowerOf2(T val)
{
    long long _val = val;
    if((_val & (-_val))-_val == 0 && _val != 0)
        return 1;
    else
        return 0;
}

  void*  create_input() {
    task_t *t = new task_t;
    fillRandom(t->inpt, width, height, 0, 255,0);
	
    /* Fill a blurr filter or some other filter of your choice*/
    for(cl_uint i=0; i < maskWidth*maskHeight; i++) t->msk[i] = 0;

    cl_float val = 1.0f/(maskWidth * 2.0f - 1.0f);

    for(cl_uint i=0; i < maskWidth; i++)
      {
          cl_uint y = maskHeight/2;
          t->msk[y*maskWidth + i] = val;
      }

	  for(cl_uint i=0; i < maskHeight; i++)
      {
          cl_uint x = maskWidth/2;
          t->msk[i*maskWidth + x] = val;
      }

	  std::cout<<"generated!"<<std::endl;
    return (void*)t;
  }



class Emitter: public ff_node {

public:
  
  Emitter(){}
      
  int svc_init(){
	return 0;
  }
    
  void * svc(void * task) {
      if (task !=NULL){
          printf("CPU done job\n");
          for(int i=0; i<height; i+=worker_width)
              for(int j=0; j<width; j+= worker_height)
                  return (void*)(new task_worker(i,j,(task_t*)task));
      }
    else
	return (void *)FF_EOS;
	}
};






class CPU_Stage: public ff_node {
public:
   
  
    int svc_init(){
        ct=0;
        return 0;
    }
  
    void * svc(void * task ) {
        //        std::cout<< ff_node::get_my_id()<<std::endl;
    if (task==NULL) return NULL;
    else{
    double cpu_start_time = get_current_time();
    task_worker *t_w =(task_worker*)task;
    cl_uint vstep = (maskWidth)/2;
    cl_uint hstep = (maskHeight)/2;

    // for each pixel in the input
    cl_uint t_i=t_w->i;
    cl_uint t_j= t_w->j;
    cl_uint m=t_i+worker_width;
    cl_uint n= t_j+ worker_height;
   // std::cout<< t_i<< std::endl;
    for(cl_uint x = t_i; x < n; x++)
        for(cl_uint y = t_j; y < m; y++)
        {
	  //find the left, right, top and bottom indices such that the indices do not go beyond image boundaires
	  cl_uint left    = (x           <  vstep) ? 0         : (x - vstep);
	  cl_uint right   = ((x + vstep) >= width) ? width - 1 : (x + vstep); 
	  cl_uint top     = (y           <  hstep) ? 0         : (y - hstep);
	  cl_uint bottom  = ((y + hstep) >= height)? height - 1: (y + hstep); 

	  //initializing wighted sum value
	  cl_float sumFX = 0;

	  for(cl_uint i = left; i <= right; ++i)
	    for(cl_uint j = top ; j <= bottom; ++j)    
	    {
	    //  printf("pair(%d,%d)\n", i,j);
		//performing wighted sum within the mask boundaries
		//cl_uint maskIndex=(i-left)*maskWidth +(j-top);
		cl_uint maskIndex = (j - (y - hstep)) * maskWidth  + (i - (x - vstep));
		cl_uint index     = j                 * width      + i;
		// printf("pair_indx(%d,%d)\n", index,maskIndex);
		//to round to the nearest integer
		// printf("par_mul(%f,%f)\n", t->inpt[index],t->msk[maskIndex]);
		sumFX += ((float)t_w->t->inpt[index] * t_w->t->msk[maskIndex]);
	    }
	  sumFX += 0.5f;
	  //verificationOutput
	    t_w->t->outpt[y*width + x] = cl_uint(sumFX);
        }

	//std::cout<<"Passed!" << std::endl;
      
    // free(t_w);
      //  free(t->outpt);
     //   free(t->msk);
     //   free(t);
       // popsignal();
    double cpu_end_time = get_current_time();
    ct += (cpu_end_time - cpu_start_time);

        return GO_ON;
     }
   }
    void svc_end(){printf("worker%d : %f\n",ff_node::get_my_id(),ct);}
private:
    double ct;
 
};


class GPU_Stage: public ff_oclNode {
   
public:
  
  GPU_Stage(){}
  
   int svc_init(){
    gtm=0;
    return 0;}
      void svc_SetUpOclObjects(cl_device_id dId){
       std::cout<< dId<< std::endl;
      // creating the context
      context = clCreateContext(NULL,1,&dId,NULL,NULL,&status);
      printStatus("CreatContext: ", status);
       /* create a CL program using the kernel source */
      std::string kernelPath = "./SimpleConvolution_Kernels_modified.cl";
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
      printf("%s\n", buff);
      //creating command quue
      //  printStatus("BuildProgram",status);
      /* The block is to move the declaration of prop closer to its use */
      cl_command_queue_properties prop = 0;
      commandQueue = clCreateCommandQueue(context, dId, prop, &status);
      
      printStatus("CreateCommandQueue",status);
    
       inputBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_uint ) * width * height, NULL, &status);
       printStatus("createbuf1",status);
       outputBuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_uint ) * width * height,NULL, &status);
       printStatus("createbuf2",status);
       maskBuffer = clCreateBuffer(context, CL_MEM_READ_ONLY , sizeof(cl_float ) * maskWidth * maskHeight,NULL, &status);
       printStatus("createbuf3",status);
    
      //cl_uint2 inputDimension = {width, height};
      //cl_uint2 maskDimension  = {maskWidth, maskHeight};
      
      /* get a kernel object handle for a kernel with the given name */
      kernel = clCreateKernel(program, "simpleConvolution", &status);
      printStatus("CreateKernel",status);
      
      /*** Set appropriate arguments to the kernel ***/
      status = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&outputBuffer);
      printStatus("setKernel1",status);
      status = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&inputBuffer);
      printStatus("setKernel2",status);
      status = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&maskBuffer);
      printStatus("setKernel3",status);
      status = clSetKernelArg(kernel, 3, sizeof(cl_uint2), (void *)&inputDimensions);
      printStatus("setKernel4",status);
      status = clSetKernelArg(kernel, 4,  sizeof(cl_uint2), (void *)&maskDimensions);
      //      clSetKernelArg(kernel, 5, 256*(sizeof(uint)), NULL);
       printStatus("setKernel5",status);
       status =clSetKernelArg(kernel, 5, 4*maskWidth*maskHeight*(sizeof(uint)), NULL);
       printStatus("setKernel6",status);
       status= clSetKernelArg(kernel, 6, maskWidth*maskHeight*(sizeof(float)), NULL);
       printStatus("setKernel7",status);              
      /* Check group size against kernelWorkGroupSize */
      status = clGetKernelWorkGroupInfo(kernel, dId,CL_KERNEL_WORK_GROUP_SIZE,sizeof(size_t),&kernelWorkGroupSize,0);
     
      printf("run once now\n");
     
  }
   bool  device_rules(cl_device_id dId){ 
       // setThreshold(4);
      size_t len;
      cl_device_type d_type;
      clGetDeviceInfo(dId, CL_DEVICE_VENDOR, 0, NULL, &len);
      char* vendor = new char[len];
      clGetDeviceInfo(dId, CL_DEVICE_VENDOR, len, vendor, NULL);
      printf(" the device vendor is %s\n", vendor);
      clGetDeviceInfo(dId, CL_DEVICE_TYPE, sizeof(cl_device_type), &(d_type), NULL);
      //  if (d_type& CL_DEVICE_TYPE_GPU) return true;
      //else return false;
        return true;
  }
   
    /* 
     * THRESHOLD is the static estimation of the maximum number of ff_node which can run on GPU. 
     * By default the value is set to 1 ff_node per GPU.
     */
    //    setThreshold(4);

   void * svc(void * task) {
      if (task==NULL){ printf("null\n"); return NULL;}
      else{
      task_t *t =(task_t*)task;
      //std::cout<<t->id<<std::endl;
      cl_int   status;
      cl_event events[2];
      size_t globalThreads[1];
      size_t localThreads[1];
      globalThreads[0] = width*height;
      //globalThreads[1]=height;
      localThreads[0]  =maskWidth*maskHeight;;
      //  localThreads[1]=worker_height;

      double gpu_start_time = get_current_time();
      // Use clEnqueueWriteBuffer() to write input array A to the device buffer bufferA
      status = clEnqueueWriteBuffer(commandQueue,inputBuffer,CL_FALSE,0,sizeof(cl_uint ) * width * height, t->inpt,0,NULL,NULL);
      // Use clEnqueueWriteBuffer() to write input array B to the device buffer bufferB
      status = clEnqueueWriteBuffer(commandQueue,maskBuffer,CL_FALSE,0,sizeof(cl_float) * maskWidth * maskHeight,t->msk,0,NULL,NULL); 
  
      //if((cl_uint)(localThreads[0]) > kernelWorkGroupSize)
      //{
//    if(!quiet)
//    {
//        std::cout << "Out of Resources!" << std::endl;
//        std::cout << "Group Size specified : "<< localThreads[0] << std::endl;
//        std::cout << "Max Group Size supported on the kernel : " << kernelWorkGroupSize <<std::endl;
//        std::cout <<"Changing the group size to " << kernelWorkGroupSize << std::endl;
//    }

      //localThreads[0] = kernelWorkGroupSize;
      //}

      //for(int iter=0; iter<10;iter++){
      /* Enqueue a kernel run call.*/
      status = clEnqueueNDRangeKernel(commandQueue,kernel,1,NULL,globalThreads,localThreads,0,NULL,&events[0]);
  //    printStatus("enqueueNDRangeKernel",status);

      /* wait for the kernel call to finish execution */
      status = clWaitForEvents(1, &events[0]);
      /* Enqueue readBuffer*/
      status = clEnqueueReadBuffer(commandQueue,outputBuffer,CL_TRUE, 0,width * height * sizeof(cl_uint),t->outpt,0,NULL,&events[1]); 
    //  printStatus("clEnqueueReadBuffer",status);
      /* Wait for the read buffer to finish execution */
      status = clWaitForEvents(1, &events[1]);
      status = clReleaseEvent(events[0]);
      status = clReleaseEvent(events[1]);
      //}
     // if(!quiet) {
    //   printArray<cl_uint >("Output", t->outpt, width, height);
      //}
    
      double gpu_end_time = get_current_time();
      gtm += (gpu_end_time - gpu_start_time);
      
      /*          std::cout<<"\n----------input---------------\n"<<std::endl;
      for(int i=0; i<width; i++)
          for(int j=0; j<height;j++){
              std::cout<<"input["<<i<<"]["<<j<<"]"<<t->inpt[i*width+j]<< "\t";
          }
      std::cout<<"\n--------------output----------\n";
      for(int i=0; i<width; i++)
          for(int j=0; j<height;j++){
              std::cout<<"output["<<i<<"]["<<j<<"]"<<t->outpt[i*width+j]<< "\t";
          }
      std::cout<<"\n-------------------mask-----\n";
      for(int i=0; i<maskWidth; i++)
          for(int j=0; j<maskHeight;j++){
              std::cout<<"mask["<<i<<"]["<<j<<"]"<<t->msk[i*maskWidth+j]<< "\t";
              }*/

              
     printf("GPU done the job!\n"); 
        free(t->inpt);
        free(t->outpt);
        free(t->msk);
        free(t);
        // popsignal();
        return GO_ON;
      //return (void *)t;
     }
   }
   
   void svc_releaseOclObjects(){
      clReleaseKernel(kernel);
      clReleaseProgram(program);
      clReleaseCommandQueue(commandQueue);
      clReleaseMemObject(inputBuffer);
      clReleaseMemObject(maskBuffer);
      clReleaseMemObject(outputBuffer);
      clReleaseContext(context);
   }
   
   void  svc_end() {
     printf("Gpu job have finished%.15f\n",gtm);
   }

private:
   unsigned int streamlen;
   size_t datasize;
   cl_context context;
   cl_program program;
   cl_command_queue commandQueue;
   cl_mem inputBuffer;
   cl_mem maskBuffer;
   cl_mem outputBuffer;
   cl_kernel kernel;
   cl_int status;
   size_t kernelWorkGroupSize;
   double gtm;
};

class ff_pipe_first_stage: public ff_node{
public:
    ff_pipe_first_stage(){};
    int svc_init(){
      first_pipe_stage_time = get_current_time();
      streamlen=0;
      return 0;
    }
    void* svc(void *){
      if(streamlen<max_strlen){
        ++streamlen;
        return create_input();
      }else return (void*)FF_EOS;

    }
    void svc_end(){
        end_pipe_stage_time = get_current_time();
        printf("time spent in the first stage of the pipeline:%f\n", end_pipe_stage_time- first_pipe_stage_time);
    }
private:
      int streamlen;
      double    first_pipe_stage_time;
    double    end_pipe_stage_time;
};

int main(int argc, char * argv[]) {
  
  
 // if (argc!=2) {
  //    std::cerr << "use: "  << argv[0] << " streamlen\n";
  //    return -1;
 // }
   /*  farm*/ 
  //create_input();
  ff_farm<> farm;//true, nworkers*nworkers*100);
    //Collector C;
    
  farm.add_collector(NULL);
  Emitter E;
  farm.add_emitter(&E); 
  std::vector<ff_node *> w;
  for(int i=0;i<nworkers;++i) 
    w.push_back(new CPU_Stage);
  farm.add_workers(w);
  
  ff_farm<> global_farm;
  global_farm.add_collector(NULL);
  std::vector<ff_node*> gw;
   gw.push_back(&farm);
   for(int i=0;i<gnworkers;++i)
         gw.push_back(new GPU_Stage);
   global_farm.add_workers(gw);
  
  ff_pipeline pipe;
  pipe.add_stage(new ff_pipe_first_stage);
  pipe.add_stage(&global_farm);
  double cpu_start_time = get_current_time();
  if (pipe.run_and_wait_end()<0/*pipe.run_and_wait_end()<0*/) {
    error("running pipeline\n");
    return -1;
  }
  global_farm.ffStats(std::cout);
  double cpu_end_time = get_current_time();
  std::cout<< "total_time:"<< cpu_end_time-cpu_start_time<<std::endl;
     
  return 0;
}

double get_current_time()
{
   static int start = 0, startu = 0;
   struct timeval tval;
   double result;

   if (gettimeofday(&tval, NULL) == -1)
      result = -1.0;
   else if(!start) {
      start = tval.tv_sec;
      startu = tval.tv_usec;
      result = 0.0;
   }
   else
      result = (double) (tval.tv_sec - start) + 1.0e-6*(tval.tv_usec - startu);

   return result;
}

