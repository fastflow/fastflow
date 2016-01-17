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
//  Version using only the ordered farm:
//    ofarm(Stage1+Stage2)

#include <opencv2/opencv.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>

using namespace ff; 
using namespace cv;

// reads frame and sends them to the next stage
struct Source : ff_node_t<cv::Mat> {
    const std::string filename;
    Source(const std::string filename):filename(filename) {}
  
    cv::Mat * svc(cv::Mat *) {
	VideoCapture cap(filename.c_str()); 
	if(!cap.isOpened())  {  
	    std::cout << "Error opening input file" << std::endl;
	    return EOS;
	} 
	for(;;) {
	    Mat * frame = new Mat();
	    if(cap.read(*frame))  ff_send_out(frame);
	    else break;
	}

    std::cout << "End of stream in input" << std::endl; 
	return EOS;
    }
}; 

// this stage applys all the filters:  the GaussianBlur filter and the Sobel one, 
// and it then sends the result to the next stage
struct Stage1 : ff_node_t<cv::Mat> {
    cv::Mat * svc(cv::Mat *frame) {
        Mat frame1;
        cv::GaussianBlur(*frame, frame1, cv::Size(0, 0), 3);
        cv::addWeighted(*frame, 1.5, frame1, -0.5, 0, *frame);
        cv::Sobel(*frame,*frame,-1,1,0,3);
        return frame;
    }
    long nframe=0;
}; 

// this stage shows the output
struct Drain: ff_node_t<cv::Mat> {
    Drain(bool ovf):outvideo(ovf) {}

    int svc_init() {
	if(outvideo) namedWindow("edges",1);
	return 0; 
    }

    cv::Mat *svc (cv::Mat * frame) {
	if(outvideo) {
	    imshow("edges", *frame);
	    waitKey(30);    
	} 
	delete frame;
	return GO_ON;
    }
protected:
    const bool outvideo; 
}; 

int main(int argc, char *argv[]) {
    //ffvideo input.mp4 filterno output nw1
    Mat edges;

    if(argc == 1) {
      std::cout << "Usage is: " << argv[0] 
                << " input_filename videooutput nw1" 
		<< std::endl; 
      return(0); 
    }
    
    // output 
    bool outvideo = false; 
    if(atoi(argv[2]) == 1) outvideo = true; 
    
    // pardegree 
    size_t nw1 = 1;
    if(argc == 4) {
      nw1 = atol(argv[3]); 
    }

    // creates an ordered farm
    ff_OFarm<cv::Mat> ofarm( [nw1]() {
            
            std::vector<std::unique_ptr<ff_node> > W; 
            for(size_t i=0; i<nw1; i++) 
                W.push_back(make_unique<Stage1>());
            return W;
            
        } ());
    
    Source source(argv[1]);
    ofarm.setEmitterF(source);
    Drain  drain(outvideo);
    ofarm.setCollectorF(drain);
    
    ffTime(START_TIME);
    // starts the pipe and waits for termination
    if (ofarm.run_and_wait_end()<0) {
        error("running pipe");
        return -1;
    }
    ffTime(STOP_TIME);

    std::cout << "Elapsed (farm(" << nw1 << "): elapsed time =" ;     
    std::cout << ffTime(GET_TIME) << " ms\n";    
    return 0;
}

