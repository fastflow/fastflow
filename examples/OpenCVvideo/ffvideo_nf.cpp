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
	    else {
            std::cout << "End of stream in input" << std::endl; 
		break;
	    }
	}
	return EOS;
    }
}; 

// this stage applys all the filters:  the GaussianBlur filter and the Sobel one, 
// and it then sends the result to the next stage
struct Stage1 : ff_node_t<cv::Mat> {
    Stage1(bool filter1, bool filter2):filter1(filter1), filter2(filter2) {}
    cv::Mat * svc(cv::Mat *frame) {
        if (!filter1 && !filter2) return frame;

        Mat frame1;
        if (filter1) {
            cv::GaussianBlur(*frame, frame1, cv::Size(0, 0), 3);
            cv::addWeighted(*frame, 1.5, frame1, -0.5, 0, *frame);
        }
        if (filter2)
            cv::Sobel(*frame,*frame,-1,1,0,3);
        return frame;
    }
    bool filter1, filter2;
}; 

// this stage shows the output
struct Drain: ff_node_t<cv::Mat> {
    Drain(bool ovf):outvideo(ovf) {}

    cv::Mat *svc (cv::Mat * frame) {
        if(outvideo) {
            imshow("Parallel", *frame);
            waitKey(30);    // why 30ms here ?
        } 
        delete frame;
        return GO_ON;
    }
protected:
    const bool outvideo; 
}; 

int main(int argc, char *argv[]) {
    Mat edges;

    if(argc != 5) {
      std::cout << "Usage is: " << argv[0] 
		<< " input_filename filterno videooutput nw1" 
		<< std::endl; 

      std::cout << "  filterno: \n";
      std::cout << "   0      : no filtering\n" 
                << "   1      : GaussianBlur filter only\n"
                << "   2      : Sobel filter only\n"
                << "   3      : both 1 and 2\n\n";
      std::cout << "  videooutput:  \n"
                << "   0      : no output\n"
                << "   1      : video output\n\n";

      std:: cout << "  nw1 : \n"
                 << "   number of workers of the farm \n\n";

      return(0); 
    }
    
    // output 
    bool outvideo = false; 
    bool filter1, filter2;
    if(atoi(argv[2]) == 1) { 
        filter1 = true;  filter2 = false; 
    }
    if(atoi(argv[2]) == 2) { 
        filter1 = false; filter2 = true;     
    }
    if(atoi(argv[2]) == 3) { 
        filter1 = true;  filter2 = true;     
    }
    
    if (atoi(argv[3]) == 1) outvideo = true; 
    
    // pardegree 
    size_t nw1 = 1;
    if(argc == 5) {
      nw1 = atol(argv[4]); 
    }
    
    // creates the pipe and adds the first stage
    ff_Pipe<> pipe(make_unique<Source>(argv[1]));
    
    // adds the second stage (sequential or ordered farm)
    if (nw1 == 1) {
        // the second stage is sequential
        pipe.add_stage(make_unique<Stage1>(filter1,filter2));
    } else {
        // the second stage is an ordered task-farm 
        // by default the ordered farm has the collector
        pipe.add_stage(make_unique<ff_OFarm<cv::Mat> >([nw1, filter1, filter2]() {

                    std::vector<std::unique_ptr<ff_node> > W; 
                    for(size_t i=0; i<nw1; i++) 
                        W.push_back(make_unique<Stage1>(filter1, filter2));
                    return W;
                    
                } ()
                ));	  
    }
    
    // adds the last stage 
    pipe.add_stage(make_unique<Drain>(outvideo));

    ffTime(START_TIME);
    // starts the pipeline and waits for termination
    if (pipe.run_and_wait_end()<0) {
	error("running pipeline");
	return -1;
    }
    ffTime(STOP_TIME);

    std::cout << "Elapsed (seq,"; 
    if(nw1 == 1)  std::cout << "seq,"; 
    else    	  std::cout << "farm(" << nw1 << "),";
    if (1)        std::cout << "seq) : elapsed time =" ;     
    std::cout << ffTime(GET_TIME) << " ms\n";

    return 0;
}

