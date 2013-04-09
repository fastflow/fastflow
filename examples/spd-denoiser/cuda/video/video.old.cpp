//
//  main.m
//  test-video
//
//  Created by Marco Aldinucci on 9/7/11.
//  Copyright 2011 Computer Science Dept. - University of Torino. All rights reserved.
//

#include <iostream>
#include <sstream>
using namespace std;
//OpenCV
#include <opencv/highgui.h>
#include <opencv/cv.h>

#define FPS 1

int main(int argc, char *argv[])
{
  /*
   * Code written by Vinz (GeckoGeek.fr)
   */
	
  char key;
  IplImage *image;
  CvCapture *capture;

  //compute frame rate
  double frame_rate = (double)1 / FPS;
		
  //open the video stream (from camera)
  capture = cvCreateCameraCapture(CV_CAP_ANY);
  if (!capture) {
    cerr << "Could not open the stream" << endl;
    exit(1);			
  }

  //print image info
  image = cvQueryFrame(capture);
  cout << "***\nImage info\ncolor channels: " << image->nChannels
       << "\ndepth: " << image->depth
       << "\nwidth: " << image->width
       << "\nheight: " << image->height
       << "\ndataOrder: " << image->dataOrder
       << "\norigin: " << image->origin
       << "\nwidthStep: " << image->widthStep
       << "\nimageSize: " << image->imageSize
       << "\nalign: " << image->align
       << "\n***" << endl;
		
  //create GUI window
  stringstream title_s;
  title_s << "Video @ " << FPS << " fps (frame rate: " << frame_rate << ")";
  string title = title_s.str();
  char *title_c = new char[title.size() + 1];
  strcpy(title_c, title.c_str());
  cvNamedWindow(title_c, CV_WINDOW_AUTOSIZE);
		
  //capture cycle
  while(key != 'q' && key != 'Q') {			
    image = cvQueryFrame(capture);
    cvShowImage(title_c, image);
    key = cvWaitKey((double)1000 / FPS);
  }

  //clean-up
  cvReleaseCapture(&capture);
  cvDestroyWindow(title_c);
  delete[] title_c;
}
