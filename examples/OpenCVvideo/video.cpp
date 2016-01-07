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
 */
/* 
 * Author: Marco Danelutto <marcod@di.unipi.it> 
 * Date:   September 2015
 * 
 */

#include <ff/utils.hpp>
#include <opencv2/opencv.hpp>

// #define SHOWTIMES
#ifdef SHOWTIMES
/* times are in millisecond */
#define TIME(t0)  ( (ff::getusec() - t0) /  1000 )
#endif

using namespace cv;

int main(int argc, char *argv[]) {
    
    if(argc == 1) {
	std::cout << "Usage is: " << argv[0] 
		  << " input_filename filterno videooutput" 
		  << std::endl; 
	return(0); 
    }
        
    // input file
    //VideoCapture cap(1); // open the default camera
    VideoCapture cap(argv[1]); 
    
    if(!cap.isOpened())  {  // check if we succeeded
	std::cerr << "Error opening input file" << std::endl;
	return -1;
    }
    std::cout << "Input file " << argv[1] << " opened" << std::endl;
    
    // filter parameters
    bool filter1=false, filter2=false;
    if(atoi(argv[2]) == 1) { 
        filter1 = true;  filter2 = false; 
        std::cout << "Applying enhnace filter only" << std::endl;
    }
    if(atoi(argv[2]) == 2) { 
        filter1 = false; filter2 = true;     
        std::cout << "Applying emboss filter only" << std::endl;
    }
    if(atoi(argv[2]) == 3) { 
        filter1 = true;  filter2 = true;     
        std::cout << "Applying both filters" << std::endl;
    }
    
    
    // output 
    bool outvideo = false; 
    if(atoi(argv[3]) == 1) outvideo = true; 
    
    Mat edges;
    if(outvideo) 
	namedWindow("edges",1);
    
    ff::ffTime(ff::START_TIME);
    int frames = 0; 
    for(;;)  {
        Mat frame1;
        Mat frame;
        
#ifdef SHOWTIMES
        unsigned long t0 = ff::getusec();
#endif
        if(cap.read(frame) == false) 
            break; 
#ifdef SHOWTIMES
        std::cout << "Read " << TIME(t0) << std::endl;
#endif
        
        frames++; 
        if(filter1) {
#ifdef SHOWTIMES
            t0 = ff::getusec();
#endif
            cv::GaussianBlur(frame, frame1, cv::Size(0, 0), 3);
            cv::addWeighted(frame, 1.5, frame1, -0.5, 0, frame);
#ifdef SHOWTIMES
            std::cout << "Filter1 " << TIME(t0) << std::endl;;
#endif
        }
        if(filter2) {
#ifdef SHOWTIMES
            t0 = ff::getusec();
#endif
            // Sobel
            Sobel(frame,frame,-1,1,0,3);
#ifdef SHOWTIMES
            std::cout << "Filter2 " << TIME(t0) << std::endl; 
#endif
        }
        
        if(outvideo) {
#ifdef SHOWTIMES
            t0 = ff::getusec();
#endif
            imshow("edges", frame);
            if(waitKey(30) >= 0) break;
            
#ifdef SHOWTIMES
            std::cout << "Show " << TIME(t0) << std::endl; 
#endif
        }

    }
    ff::ffTime(ff::STOP_TIME);
    std::cout << "Elapsed time is " << ff::ffTime(ff::GET_TIME) << " ms\n";
    std::cout << "Average time per frame " << (ff::ffTime(ff::GET_TIME) / frames) <<  " ms\n";
    std::cout << "(with " << frames << " frames)" << std::endl;
    
    // the camera will be deinitialized automatically in VideoCapture destructor
    return 0;
}
