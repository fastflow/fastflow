//
//  denoiser
//
//  Created by Maurizio Drocco on 2/15/13.
//  Copyright 2013 Computer Science Dept. - University of Torino. All rights reserved.
//

//default mode: BITMAP (single picture)
#if (not defined FROMCAMERA) and (not defined FROMFILE) and (not defined BITMAP) and (not defined BITMAPCOLOR)
#define BITMAP
#endif

//default mode: FF
#if (not defined SPD_OCL) and (not defined SPD_FF)
#define SPD_FF
#endif

#ifdef SPD_OCL
#define MODE "OpenCL"
#define OCL_SETUP_TIME 0
#include "pipe_ocl.hpp"
#endif
#ifdef SPD_FF
#define MODE "FF"
#include "pipe_ff.hpp"
#endif

#define MAXINFLIGHT 4
#define FPS_DEFAULT 10
#ifdef TIME
#define RAN 8 //recent framerate size
#endif

#include <iostream>
using namespace std;

//FF
#include <ff/pipeline.hpp>
using namespace ff;

//openCV
#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <opencv2/opencv.hpp>

#include "defs.h"
#include "parameters.h"
#include "cast.hpp"
#include "noiser.hpp"
#include "utils.hpp"
#include "capture_utils.hpp"



int main(int argc, char *argv[]) {
  int actualnoise;
  char key;
  IplImage *image, *filteredframe, *imageresult;
  unsigned int width, height;
  CvCapture *capture;
  int count=0, n_restored=0, inflight=0;
  float noise_avg = 0, cycles_avg = 0;
  void *result = NULL;

  //parse command-line arguments
  string fname, out_fname;
  float alfa, beta;
  unsigned int w_max, max_cycles, noise;
  bool verbose, show_enabled, fixed_cycles, add_noise, user_out_fname;
  arguments args;
  get_arguments(argv, argc, args);
  fname = args.fname;
  alfa = args.alfa;
  beta = args.beta;
  w_max = args.w_max;
  max_cycles = args.max_cycles;
  fixed_cycles = args.fixed_cycles;
  user_out_fname = args.user_out_fname;
  if(user_out_fname)
    out_fname = args.out_fname;
  verbose = args.verbose;
  show_enabled = args.show_enabled;
  add_noise = args.noise > 0;
  noise = args.noise;

  //build output fname
  string prefix;
  if(user_out_fname)
    prefix = out_fname.substr(0, out_fname.length() - 4);
  else {
    prefix.append("RESTORED_");
#ifdef FROMCAMERA
    prefix.append("fromcamera");
#else
    string trunked_fname = get_fname(fname);
    prefix.append(trunked_fname.substr(0, trunked_fname.length() - 4));
#endif
    prefix.append("_" + string(MODE));
  }

  //verbose headers
  if(verbose) {
    cout << "*** This is Salt & Pepper denoiser" << endl
	 << "mode: (" << MODE << ") flat "
	 << "| termination: average " << endl
	 << "control-window size: " << w_max << endl
	 << "alpha = " << alfa << "; beta = " << beta << endl
	 << "max number of cycles = " << max_cycles << endl;
    if(fixed_cycles)
      cout << "number of cycles fixed to " << max_cycles <<  endl;
    if(add_noise)
      cout << "will add " << noise << "% of noise" << endl;
#ifndef FROMCAMERA
    cout << "Input: " << fname << endl
	 << "Output: " << prefix << "." << OUTFILE_EXT << endl << "---" << endl;
#endif
  }



#ifdef TIME
  long int timer, time_throughput, time_input = 0, time_output = 0;
  long int start_usec, time_usec, prev_time;
  long int time_usec_ra[RAN], ra; // running average framerate
  memset (time_usec_ra, 0, sizeof(long)*RAN);
  int ra_i=0;
  long int svc_time_detect=0, svc_time_denoise=0;
#endif

  if(show_enabled) {
    cvNamedWindow("Original", CV_WINDOW_AUTOSIZE);
    cvNamedWindow("Restored", CV_WINDOW_AUTOSIZE);
  }

  //*** setup capture device
#ifdef FROMCAMERA
  int c_size[4][2] = {{320,240},{426,320},{640,480},{1280,960}};
  int c_size_i = 0;							
  capture = cvCreateCameraCapture(CV_CAP_ANY);
  if (!capture) {
    printf("Error: Cannot open the webcam !\n");
    return 1;
  }
  width = c_size[c_size_i][0];
  height = c_size[c_size_i][1];
  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH,width);
  cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT,height);
#endif

#ifdef FROMFILE
  capture = cvCaptureFromFile(fname.c_str());
  double fps = cvGetCaptureProperty(capture, CV_CAP_PROP_FPS);
  width = (unsigned int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH);
  height = (unsigned int) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT);
  unsigned long n_frames = (unsigned long) cvGetCaptureProperty(capture,CV_CAP_PROP_FRAME_COUNT);
  int fourcc = (int)cvGetCaptureProperty(capture, CV_CAP_PROP_FOURCC);

  if(verbose)
    cout << n_frames << " frames ( " << width << " x " << height << "  @ " << fps << " fps)" << endl;
  actualnoise = noise;
#endif

  //set output stream
#ifdef FROMFILE
  CvVideoWriter* writer = cvCreateVideoWriter((prefix + "." + OUTFILE_EXT).c_str(),
					      CV_FOURCC('X','V','I','D'),
					      fps, cvSize((int)width,(int)height),1);
#endif

  //read picture (single-bitmap mode)
#ifdef BITMAP
  image = cvLoadImage(fname.c_str());
  CvSize size = cvGetSize(image);
  width = (unsigned int)image->width;
  height = (unsigned int)image->height;
  cerr << "width " << width << " height " << height << "\n";
  if(show_enabled)
    cvShowImage( "Original", image);
  ++count;
#endif


#ifdef BITMAPCOLOR

 image = cvLoadImage(fname.c_str());
 CvSize size = cvGetSize(image);
 width = (unsigned int)image->width;
 height = (unsigned int)image->height;
 cerr << "orig width " << width << " height " << height <<  "  depth " << image->depth << "\n";
 IplImage* R = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
 IplImage* G = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
 IplImage* B = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
 cvSplit(image, R, G, B, NULL);

 if(show_enabled){
   cvShowImage( "Original", image);
//   cvShowImage( "R", R);
//   cvShowImage( "G", G);
//   cvShowImage( "B", B);
 }
#endif

  //*** set the pipeline
  ff_pipeline pipe(true);
  pipe_components_t comp;
  unsigned int par_degree;
#ifdef SPD_OCL
  par_degree = 1; //gpu count
#endif
#ifdef SPD_FF
  par_degree = 4; //core count
#endif
  setup_pipe(pipe, comp, w_max, alfa, beta, par_degree);

  //run
  pipe.run();
#ifdef SPD_OCL
  //sleep while ocl init devices (to be removed?)
  cerr << "sleep " << OCL_SETUP_TIME << " s (device setup)... " << flush;
  sleep(OCL_SETUP_TIME);
#endif
  cerr << "go" << endl;
  
#ifdef TIME
   start_usec = get_usec_from(0);
   time_usec = 0;
   time_throughput = start_usec;
#endif



  //*** go
  bool go = true;
  while(go) {
#ifdef TIME
    timer = get_usec_from(0);
#endif

    //read picture (single-bitmap COLOR mode)
#ifdef BITMAPCOLOR
   ++count;

//  filteredframe = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);

  if(count==1){
	 // cvCopy(R, filteredframe, NULL);
	  filteredframe = R;
  }else
	  if(count==2){
//		  cvCopy(G, filteredframe, NULL);
		  filteredframe = G;
	  }else
		  if(count==3){
//			  cvCopy(B, filteredframe, NULL);
			  filteredframe = B;
		  }
  #endif

#ifdef FROMCAMERA
    //dynamic reconfiguration (to be removed)
    noise = adjust(capture, key, noise, c_size, &c_size_i, &height, &width);
    actualnoise = (std::max)((long)0, (std::min)(noise/2 + (noise/2)*random()/RAND_MAX, (long)255));
    if(verbose)
      cout << "actual noise: " << actualnoise << " %" << endl;
#endif

    //*** Get an image
#if defined(FROMFILE) || defined(FROMCAMERA)
    if(!(image = cvQueryFrame(capture))) {
      cerr << "Can't read from input stream" << endl;
      break;
    }

    ++count;
    if(verbose)
      cout << "Read frame n. " << count
#ifdef FROMFILE
	 << " / " << n_frames
#endif
	 << endl;

#endif //FROMFILE || FROMCAMERA

#if (not defined BITMAPCOLOR)

    //*** input filter (add noise)
    filteredframe = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1); 
    cvCvtColor(image, filteredframe, CV_RGB2GRAY);
    if(add_noise)
    	addSaltandPepperNoiseDEPTH_8U(filteredframe, actualnoise);
#endif
    //*** offload
    pipe.offload(new task_frame(count, Ipl2uchar<pixel_t>(filteredframe), height, width, max_cycles, fixed_cycles));
    ++inflight;

#ifdef TIME
    time_input += get_usec_from(timer);
#endif

    //*** get some restored frames
    do {
      while (pipe.load_result_nb(&result)) {
	++n_restored;
	--inflight;
	task_frame * rt = (task_frame *)result;

#ifdef TIME
	timer = get_usec_from(0);
	prev_time = time_usec;
	time_usec = get_usec_from(start_usec);
	time_usec_ra[ra_i] = time_usec - prev_time;
	++ra_i %= RAN;
	if(n_restored >= RAN) {
	  ra = 0;
	  for (int i=0; i<RAN; ra+=time_usec_ra[i++]);
	  if(verbose)
	    cerr << "Recent framerate " << ra_i << " " << 1e06 * (float) RAN / ra
		 << " Global (avg): " << (1e06 * (float)n_restored / time_usec) << " fps" << endl;
	}
	svc_time_detect += rt->svc_time_detect;
	svc_time_denoise += rt->svc_time_denoise;
#endif

	if(verbose)
	  cout << "Restored frame n. " << n_restored
	       << " / " << count << " (" << rt->n_noisy << " noisy, " << rt->cycles << " cycles)" << endl;
	IplImage *imageresult = uchar2Ipl<pixel_t>(rt->out, height, width);
	noise_avg += ((float)(rt->n_noisy) / (rt->height * rt->width));
	cycles_avg += rt->cycles;
	delete rt;
	if(show_enabled)
	  cvShowImage( "Restored", imageresult);

#ifdef FROMFILE
	//*** write the restored frame
	IplImage * colorimageresult = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3); 
	cvCvtColor(imageresult, colorimageresult, CV_GRAY2RGB);
	cvWriteFrame(writer,colorimageresult);
	cvReleaseImage(&colorimageresult);
	cout << "progress: " << (float)n_restored / n_frames * 100 << " %" << "\r" << flush;
	if(verbose)
	  cout << endl;
#endif
#ifdef BITMAP
	//*** write output bitmap
	cvSaveImage((prefix + "." + OUTFILE_EXT).c_str(), imageresult);
#endif
	cvReleaseImage(&imageresult);
#ifdef TIME
	time_output += get_usec_from(timer);
#endif
      }

      usleep(20); //suspend the waiting thread
    } while (inflight > MAXINFLIGHT);

#if (not defined BITMAP) and (not defined BITMAPCOLOR)
    //show the most recent noisy frame
    if(show_enabled)
      cvShowImage("Original", filteredframe);
#endif

    cvReleaseImage(&filteredframe);

#ifdef BITMAPCOLOR
    if(count==3)
      break;
#endif

#ifdef BITMAP
    break;
#else // = CAMERA + FILE
    key = cvWaitKey(10);
    go = (key != 'q' && key != 'Q');
#ifdef FROMFILE
    go &= count < n_frames;
#endif
#endif // BITMAP

  } //end loop over frames

  cerr << "end of input" << endl;

  // join all threads (collect remaining restored frames) 
  pipe.offload((void *)FF_EOS);
  cerr << "offloaded EOS" << endl;

  //could crash - START
  cerr << "collect remaining " << inflight << " frames" << endl;
  while(inflight > 0) {
    while (pipe.load_result(&result)) {
      ++n_restored;
      --inflight;
      task_frame * rt = (task_frame *)result;
      
#ifdef TIME
      timer = get_usec_from(0);
      svc_time_detect += rt->svc_time_detect;
      svc_time_denoise += rt->svc_time_denoise;
#endif

      if(verbose)
	cout << "Restored frame n. " << n_restored
	     << " / " << count << " (" << rt->n_noisy << " noisy, " << rt->cycles << " cycles)" << endl;
      IplImage *imageresult = uchar2Ipl(rt->out, height, width);
      //IplImage *imageresult = uchar2Ipl(rt->in, height, width);
      noise_avg += ((float)(rt->n_noisy) / (rt->height * rt->width));
      cycles_avg += rt->cycles;
      delete rt;
#ifdef BITMAPCOLOR

	  if(n_restored==1){
			R = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
			cvCopy(imageresult, R, NULL);
	  }else
		  if(n_restored==2){
			  G = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
			  cvCopy(imageresult, G, NULL);
		  }else
			  if(n_restored==3){ // Last frame restored => merge and write output file
				  B = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 1);
				  cvCopy(imageresult, B, NULL);

				  cvMerge(R, G, B, NULL, image);
				  cvReleaseImage(&R);
				  cvReleaseImage(&G);
				  cvReleaseImage(&B);

				  //*** write output bitmap
				  cvSaveImage((prefix + "." + OUTFILE_EXT).c_str(), image);

				  if(show_enabled)
			    	  cvShowImage( "Restored", image);
			  }
#endif

#if (not defined BITMAPCOLOR)
      if(show_enabled)
    	  cvShowImage( "Restored", imageresult);
#endif

#ifdef FROMFILE
	//*** write the restored frame
	IplImage * colorimageresult = cvCreateImage(cvSize(width, height), IPL_DEPTH_8U, 3); 
	cvCvtColor(imageresult, colorimageresult, CV_GRAY2RGB);
	cvWriteFrame(writer,colorimageresult);
	cvReleaseImage(&colorimageresult);
#endif
#ifdef BITMAP
	//*** write output bitmap
	cvSaveImage((prefix + "." + OUTFILE_EXT).c_str(), imageresult);
#endif
      cvReleaseImage(&imageresult);
      
#ifdef TIME
      time_output += get_usec_from(timer);
#endif
      //if(inflight == 0) break; //uncomment if use pipe.load_result_nb()
    }
    usleep(20); //suspend the waiting thread
  };
  //could crash - END

  if (pipe.wait() < 0) {
    error("error waiting pipe\n");
    return -1;
  }

  cerr << "pipeline closed" << endl;

  cerr << "avg. noise: " << noise_avg*100/n_restored << " %" << endl;
  cerr << "avg. cycles: " << cycles_avg/n_restored << endl;
#ifdef TIME
  cerr << "avg. input time: " << (float)time_input/1e03/n_restored << " ms" << endl;
  cerr << "avg. output time: " << (float)time_output/1e03/n_restored << " ms" << endl;
  cerr << "avg. svc time detect: " << (float)svc_time_detect/1e03/n_restored << " ms" << endl;
  cerr << "avg. svc time denoise: " << (float)svc_time_denoise/1e03/n_restored << " ms" << endl;
  unsigned long working_time = get_usec_from(time_throughput);
  cerr << "throughput: " << (float)n_restored/(working_time/1e06) << " fps"
       << " (" << n_restored << " frames in " << (float)working_time/1e06 << " s)" << endl;
#endif

#if (defined FROMCAMERA) || (defined FROMFILE)
  cvReleaseCapture(&capture);
#endif
#ifdef FROMFILE
  cvReleaseVideoWriter(&writer);
#endif

#if defined (BITMAP) || defined (BITMAPCOLOR)
  if(show_enabled)
    cvWaitKey(WAIT_FOR * 1000);
#endif

  if(show_enabled) {
    cvDestroyWindow("Original");
    cvDestroyWindow("Restored");
  }

  //clean-up
  clean_pipe(comp);

  cout << "\ndone" << endl;
  return 0;
}
