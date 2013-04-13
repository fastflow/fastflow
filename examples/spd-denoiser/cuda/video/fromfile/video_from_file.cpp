//
//  test-video
//
//  Created by Marco Aldinucci on 9/7/11.
//  Copyright 2011 Computer Science Dept. - University of Torino. All rights reserved.
//

#define MAXINFLIGHT 4
#define FROMCAMERA 1

//#include <stdio.h>
#include <iostream>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <noiser.hpp>
#include "videoTaskTypes.hpp"
#include "taskTypes.hpp"
#include "noise_detection.hpp"

//#include "ff_accel.hpp"
//#include "pow_table.hpp"
#include "fuy.hpp"
#include "convergence.hpp"
#ifdef CUDA
#include "denoise_cuda.hpp"
#endif
#include "utils.h"
using namespace ff;
//#define FPS 25


/// Begin pipeline accelerator
template <typename T>
class Detect: public ff_node {
public:
  
  Detect(unsigned int w_max): w_max(w_max) {}

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    //cerr << "Detect stage img width " << t->bmp->width() <<  " height " << t->bmp->height() << "\n";
#ifdef TIME
    long t_det = get_usec_from(0);
#endif
    t->the_noisy_set = new vector<noisy<T> >;
    t->the_noisy_set->reserve((t->bmp->height())*(t->bmp->width())/10);
    Bitmap_ctrl<T> bmp_ctrl(*(t->bmp)); // useless - remove
    find_noisy_partial<T>(*(t->bmp), bmp_ctrl, w_max, *(t->the_noisy_set), 0, t->bmp->height()-1, t->bmp->width());
#ifdef TIME
    t_det = get_usec_from(t_det)/1000;
    cerr << "Detect Time :" << t_det << " (ms) " << "Noisy pixels: " << t->the_noisy_set->size() << endl;
#endif
    return task;
  }

private:
  unsigned int w_max;

};

template <typename T>
class Denoise: public ff_node {
public:
  Denoise(double alpha):alpha(alpha) {
    pow_table_alfa = new pow_table(alpha);
  }

  void * svc(void * task) {
    noisy_img_task<T> *t = (noisy_img_task<T> *)task;
    vector<noisy<T> > &set = *(t->the_noisy_set);
    vector<grayscale> diff(set.size(),0);
    int cur_residual, old_residual = 0;
#ifdef TIME
    int cycles = 0;
    long t_rec = get_usec_from(0);
#endif
    bool fix = false;
    while (!fix) {
      cerr << "start cycle ... " << flush;
      old_residual = cur_residual;
      for(unsigned int i=0; i<set.size(); ++i)
	fuy(*(t->bmp), set[i], 0, t->bmp->width(), t->bmp->height(), 1.3, 5, i/*, &mypow*/, *pow_table_alfa);
      cur_residual = reduce_residual<grayscale>(*(t->bmp), set, diff);
#ifdef TIME
      ++cycles;
#endif
      fix = (_ABS(old_residual - cur_residual) > 0);
      cerr << "end" << endl;
    }
    //t->the_noisy_set->clear();
    delete t->the_noisy_set;
    t->the_noisy_set = NULL;
#ifdef TIME
    t_rec = get_usec_from(t_rec)/1000;
    cerr << "Denoising Time :" << t_rec << " (ms) Cycles " << cycles << "\n";
#endif
    
    return task;
  }

private:
  double alpha;
  pow_table *pow_table_alfa;
  //float current_residual, old_residual, delta_residual;
};


void onMouse(int event, int x, int y, int flags, void* mousenoise) {
  if (event==CV_EVENT_MOUSEMOVE) {
    (*(int *)mousenoise)+=10;
  }
}


int main(int argc, char *argv[]) {

  /*
    CvCapture *capture;

    capture = cvCreateCameraCapture(CV_CAP_ANY);
    cerr << "Init video done\n";
    if (!capture) {			
    cerr << "Cannot open video device \n";
    return 1;
    }
    cvNamedWindow("Window", CV_WINDOW_AUTOSIZE);
  
    ff_pipeline pipe;
    cerr << "starting\n";
    pipe.add_stage(new Capture(capture));
    pipe.add_stage(new Show(capture));	

    if (pipe.run_and_wait_end()<0) {
    error("running tctfb pipeline\n");
    return -1;
    }
    return 0;
    /*
    /*
    cvReleaseCapture(&capture);
    cvDestroyWindow("Window");
    }
  */
 
  int noise=15,mousenoise=0,actualnoise; 

  ff_pipeline pipe(true);
  pipe.add_stage(new Detect<grayscale>(75 /* w_max */));
#ifdef CUDA
  pipe.add_stage(new Denoise_cuda<grayscale>(1.3 /*alpha */, 5 /* beta */));
#else
  pipe.add_stage(new Denoise<grayscale>(1.3 /*alpha */));
#endif
  pipe.run();
  /*
    ff_farm<> farm(true);
    std::vector<ff_node *> w;
    for(int i=0;i<1;++i) w.push_back(new Filter);
    farm.add_workers(w);
    farm.add_collector(new Collector);
    farm.run();
  */
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
    cout << "Recent framerate " << ra_i << " " << 1000000 * (float) RAN / ra << " Total (avg): " << (1000000 * (float)++frames / time_usec) << endl;
    ++inflight;
    // --- input filter - simulate noisy image
    filteredframe = cvCreateImage(cvSize(image->width, image->height), IPL_DEPTH_8U, 1); 
    cvCvtColor(image, filteredframe, CV_BGR2GRAY);
    addSaltandPepperNoiseDEPTH_8U(filteredframe, actualnoise);
    // ---
    //imageclone = (IplImage **) malloc(sizeof(void *));
    //*imageclone = cvCloneImage(image);
    // Create bmp for denoising
    noisy_img_task<grayscale> * nt = new noisy_img_task<grayscale>(new Bitmap_cv<grayscale>(filteredframe), NULL); 
	 
    pipe.offload(nt);
    //key = cvWaitKey(1);
    //farm.offload(*imageclone);
    //cerr << "Image " << image << "\n";
    // Convert grayscale (for salt-and-pepper)
    //IplImage *grayscale = cvCreateImage( cvSize(image->width, image->height), IPL_DEPTH_8U, 1);
    //cvCvtColor(image, grayscale, CV_BGR2GRAY);
    //farm.load_result(&result);

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
