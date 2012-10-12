//
//  test-video
//
//  Created by Marco Aldinucci on 9/7/11.
//  Copyright 2011 Computer Science Dept. - University of Torino. All rights reserved.
//

#define MAXINFLIGHT 4
#define FROMCAMERA 1
#define W_MAX 25 //max detection-window size
//#include <stdio.h>
#include <iostream>
#include <ff/pipeline.hpp>
#ifndef SEQ_DETECTION
#include <ff/farm.hpp>
#endif
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <noiser.hpp>
#include "videoTaskTypes.hpp"
//#include "taskTypes.hpp"
#include "noise_detection.hpp"
#include "ff_accel_video.hpp"

//#include "ff_accel.hpp"
//#include "pow_table.hpp"
#include "fuy.hpp"
#include "convergence.hpp"
#ifdef FF_WITH_CUDA
#include "denoise_cuda.hpp"
#endif
#include "utils.hpp"
using namespace ff;
//#define FPS 25

void onMouse(int event, int x, int y, int flags, void* mousenoise) {
  if (event==CV_EVENT_MOUSEMOVE) {
    (*(int *)mousenoise)+=10;
  }
}



int main(int argc, char *argv[]) {
 
  int noise=15, mousenoise=0, actualnoise; 
  char key = 'i';
  IplImage *image;	   
  CvCapture *capture;
  int inflight=0;
  IplImage **imageclone, *frame, *filteredframe, *imageresult;
  void * result=NULL;
  long int start_usec = get_usec_from(0), time_usec, prev_time = start_usec;
#define RAN 8
  long int time_usec_ra[RAN],ra; // running average framerate
  memset (time_usec_ra, 0, sizeof(long)*RAN);
  int ra_i=0;
  unsigned long frames = 0;



  //build the pipeline
  ff_pipeline pipe(true);

  //detection stage
#ifdef SEQ_DETECTION
  pipe.add_stage(new Detect<grayscale>(W_MAX));
#else
#define N_DETECTION_WORKERS 2
  ff::ff_farm<> farm_detection; //accelerator set
  std::vector<ff::ff_node *> w_detectors;
  for(unsigned int i=0; i<N_DETECTION_WORKERS; ++i)
    w_detectors.push_back(new FFW_detect<grayscale>(W_MAX));
  unsigned int n_sets = 2 * N_DETECTION_WORKERS;
  farm_detection.add_emitter(new FFE_detect<grayscale>(n_sets));
  farm_detection.add_workers(w_detectors);
  farm_detection.add_collector(new FFC_detect<grayscale>(n_sets));
  pipe.add_stage(&farm_detection);
#endif

  //denoising stage
#ifdef FF_WITH_CUDA
  pipe.add_stage(new Denoise_cuda<grayscale>(1.3 /*alpha */, 5 /* beta */));
#else
  pipe.add_stage(new Denoise<grayscale>(1.3 /*alpha */, 5 /* beta */));
#endif

  //run
  pipe.run();



#ifdef FROMCAMERA  
  int c_size[4][2] = {{320,240},{426,320},{640,480},{1280,960}};
  int c_size_i = 0;							
  capture = cvCreateCameraCapture(CV_CAP_ANY);
  if (!capture) {
    printf("Error: Cannot open the webcam !\n");
    return 1;
  }
  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,c_size[c_size_i][0]);
  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,c_size[c_size_i][1]);
  cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
  cvSetMouseCallback("Original", &onMouse, (void*) &mousenoise);
  cvNamedWindow("Restored", CV_WINDOW_AUTOSIZE);
  
  
  
  while(key != 'q' && key != 'Q') {
    if (key=='+' && noise < 85) {
      noise +=10;
      cout << "Noise " << noise << "%\n";
    }
    if (key=='-' && noise > 15) {
      noise -=10;
      cout << "Noise " << noise << "%\n";
    }
    if ((key=='l') && (c_size_i<4)) {
      c_size_i++;
      cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,c_size[c_size_i][0]);
      cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,c_size[c_size_i][1]);
    }
    if ((key=='s') && (c_size_i>0)) {
      c_size_i--;
      cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,c_size[c_size_i][0]);
      cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,c_size[c_size_i][1]);
    }
    actualnoise = noise/2 + (noise/2)*random()/RAND_MAX;
    actualnoise = noise+mousenoise;
    if (mousenoise>10) mousenoise-=5;
    if (noise>85) noise=85;
    if (noise<0) noise=0;
    cout << "ActualNoise " << actualnoise << "%\n";

#else 
    capture = cvCreateFileCapture("/Users/aldinuc/Desktop/1970.avi");
    double fps = cvGetCaptureProperty(capture,CV_CAP_PROP_FPS);
    int width = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH);
    int height = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT); 
    //int fourcc = (int) cvGetCaptureProperty(capture,CV_CAP_PROP_FOURCC);
    cerr << " fps " << fps << " width " << width << " height " << height << "\n";
    CvVideoWriter* writer =  cvCreateVideoWriter("/Users/aldinuc/Desktop/1970-r.avi", CV_FOURCC('D','I','V','X'), fps, cvSize(width,height),1);
    int count = 0;
    while(count++ < 2000) {
      cerr << "Frame n. " << count << "\n";
#endif

      // Get an image
      image = cvQueryFrame(capture);
      prev_time = time_usec;
      time_usec = get_usec_from(start_usec);
      time_usec_ra[ra_i] = time_usec - prev_time;
      ++ra_i %= RAN;
      ra = 0;
      for (int i=0; i<RAN; ra+=time_usec_ra[i++]);
      cout << "Framerate (fps): recent " << ra_i << " " << 1000000 * (float) RAN / ra
	   << ", global " << (1000000 * (float)++frames / time_usec) << endl;
      ++inflight;

      // --- input filter - simulate noisy image
      filteredframe = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1); 
      cvCvtColor(image, filteredframe, CV_BGR2GRAY);
      addSaltandPepperNoiseDEPTH_8U(filteredframe, actualnoise);

      // Create bmp for denoising
      noisy_img_task<grayscale> * nt = new noisy_img_task<grayscale>(new Bitmap_cv<grayscale>(filteredframe));

      pipe.offload(nt); //send a task

      //wait for denoised pictures
      do {
	while (pipe.load_result_nb(&result)) {
	  --inflight;
	  noisy_img_task<grayscale> * rt = (noisy_img_task<grayscale> *) result;
	  Bitmap_cv<grayscale> *bcv = static_cast<Bitmap_cv<grayscale>*>(rt->bmp);
	  IplImage *imageresult = bcv->createImg_cv(IPL_DEPTH_8U,1);
	  delete rt->bmp;
	  delete rt;
#ifdef FROMCAMERA
	  cvShowImage( "Restored", imageresult);
#else
	  IplImage * colorimageresult = cvCreateImage(cvSize(imageresult->width, imageresult->height), IPL_DEPTH_8U, 3);
	  cvCvtColor(imageresult, colorimageresult, CV_GRAY2BGR);
	  cvWriteFrame(writer,colorimageresult);
	  cvReleaseImage(&colorimageresult);
#endif
	  cvReleaseImage(&imageresult);
	} 
	usleep(20);
      } while (inflight > MAXINFLIGHT);
      key = cvWaitKey(10);
#ifdef FROMCAMERA
      cvShowImage("Original", filteredframe);
#endif	
      cvReleaseImage(&filteredframe);
    }


    cvReleaseCapture(&capture);
#ifdef FROMCAMERA
    cvDestroyWindow("Original");
    cvDestroyWindow("Restored");
#else
    cvReleaseVideoWriter(&writer);
#endif
    // join all threads
    pipe.offload((void *)FF_EOS);
    while (pipe.load_result(&result));

    if (pipe.wait() < 0) {
      error("error waiting pipe\n");
      return -1;
    }
  
  }
