#ifndef _OCV_VIDEO_OUTPUT_HPP_
#define _OCV_VIDEO_OUTPUT_HPP_

#include <ocv/cast.hpp>
#include <Output.hpp>

#include <opencv/highgui.h>
#include <opencv2/opencv.hpp>
#include <string>

/*!
 * \class VideoOutputOCV
 *
 * \brief Writes frames to a video stream using OpenCV
 *
 * This class writes frames to a video stream using OpenCV.
 */
class VideoOutputOCV: public Output {
public:

  VideoOutputOCV(string out_fname, double fps, int height, int width, bool show) :
      Output(height, width), show_enabled(show) {
    //get frame info
	name = out_fname;
    int fourcc = CV_FOURCC('X', 'V', 'I', 'D'); // static_cast<int>(capture.get(CV_CAP_PROP_FOURCC));
    //out_fname.append(in_fname.substr(in_fname.length() - 3, in_fname.length())); //same extension as input file
    out_fname.append(".avi");
    outputVideo.open(out_fname, fourcc, fps, cv::Size((int) width, (int) height), false);

    if (!outputVideo.isOpened()) {
      std::cout << "!!! Output video could not be opened " << out_fname << std::endl;
    }
  }

  void writeFrame(unsigned char *frame) {
	  //write
	  cv::Mat outFrame = cv::Mat(height, width, CV_8U, frame, cv::Mat::AUTO_STEP); // convert uc* to Mat

	  if(show_enabled){
		  imshow(name, outFrame);
		  waitKey(1);
	  }
	  outputVideo << outFrame;
  }

private:
  string name;
  bool show_enabled;
  cv::VideoWriter outputVideo;
};
#endif
