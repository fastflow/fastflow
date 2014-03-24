#ifndef _OCV_BITMAP_OUTPUT_HPP_
#define _OCV_BITMAP_OUTPUT_HPP_

#include <Output.hpp>

#include <opencv2/opencv.hpp>
#include <string>

/*!
 * \class BitmapOutputOCV
 *
 * \brief Writes bitmap to a file using OpenCV
 *
 */
class BitmapOutputOCV: public Output {
public:
  BitmapOutputOCV(string out_fname, int height, int width) :
      Output(height, width), fname(out_fname) {
    fname.append(".bmp");
  }
  ;

  void writeFrame(unsigned char *frame) {
    imwrite(fname, cv::Mat(height, width, CV_8U, frame, cv::Mat::AUTO_STEP));
  }

private:
  string fname;
};
#endif
