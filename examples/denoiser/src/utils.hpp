/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/*
 *
 *  Authors:
 *    Maurizio Drocco
 *    Guilherme Peretti Pezzi 
 *  Contributors:  
 *    Marco Aldinucci
 *    Massimo Torquati
 *
 *  First version: February 2014
 */

#ifndef UTILS_HPP_
#define UTILS_HPP_

#include <opencv2/opencv.hpp>
#include <string>
using namespace std;
using namespace cv;

//#include <ff/platforms/platform.h>

//default mode: FF
//#if (not defined SPD_OCL) and (not defined SPD_FF)
//#define SPD_FF
//#endif

#define _ABS(a)	   (((a) < 0) ? -(a) : (a))
#define _MAX(a, b) (((a) > (b)) ? (a) : (b))
#define _MIN(a, b) (((a) < (b)) ? (a) : (b))

//extract the filename from a path
string get_fname(string &path) {
  size_t p = path.find_last_of("/") + 1;
  return path.substr(p, path.length() - p);
}

//check if the file is bitmap or video (by extension)
bool isBitmap(string fname) {
  size_t p = fname.find_last_of(".") + 1;
  return strcasecmp(fname.substr(p, 3).data(), "bmp") == 0; // case insensitive test
}

#include <sys/time.h>
//funzioni per le misure di tempo
 unsigned long get_usec_from(unsigned long s) {
   struct timeval tv;
   gettimeofday(&tv, NULL);
   return (unsigned long)(tv.tv_sec * 1e06 + tv.tv_usec) - s;
 }

#ifdef SPD_OCL
#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/opencl.h>
#endif

const char* oclErrorString(cl_int error)
{
  static const char* errorString[] = {
    "CL_SUCCESS",
    "CL_DEVICE_NOT_FOUND",
    "CL_DEVICE_NOT_AVAILABLE",
    "CL_COMPILER_NOT_AVAILABLE",
    "CL_MEM_OBJECT_ALLOCATION_FAILURE",
    "CL_OUT_OF_RESOURCES",
    "CL_OUT_OF_HOST_MEMORY",
    "CL_PROFILING_INFO_NOT_AVAILABLE",
    "CL_MEM_COPY_OVERLAP",
    "CL_IMAGE_FORMAT_MISMATCH",
    "CL_IMAGE_FORMAT_NOT_SUPPORTED",
    "CL_BUILD_PROGRAM_FAILURE",
    "CL_MAP_FAILURE",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "",
    "CL_INVALID_VALUE",
    "CL_INVALID_DEVICE_TYPE",
    "CL_INVALID_PLATFORM",
    "CL_INVALID_DEVICE",
    "CL_INVALID_CONTEXT",
    "CL_INVALID_QUEUE_PROPERTIES",
    "CL_INVALID_COMMAND_QUEUE",
    "CL_INVALID_HOST_PTR",
    "CL_INVALID_MEM_OBJECT",
    "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR",
    "CL_INVALID_IMAGE_SIZE",
    "CL_INVALID_SAMPLER",
    "CL_INVALID_BINARY",
    "CL_INVALID_BUILD_OPTIONS",
    "CL_INVALID_PROGRAM",
    "CL_INVALID_PROGRAM_EXECUTABLE",
    "CL_INVALID_KERNEL_NAME",
    "CL_INVALID_KERNEL_DEFINITION",
    "CL_INVALID_KERNEL",
    "CL_INVALID_ARG_INDEX",
    "CL_INVALID_ARG_VALUE",
    "CL_INVALID_ARG_SIZE",
    "CL_INVALID_KERNEL_ARGS",
    "CL_INVALID_WORK_DIMENSION",
    "CL_INVALID_WORK_GROUP_SIZE",
    "CL_INVALID_WORK_ITEM_SIZE",
    "CL_INVALID_GLOBAL_OFFSET",
    "CL_INVALID_EVENT_WAIT_LIST",
    "CL_INVALID_EVENT",
    "CL_INVALID_OPERATION",
    "CL_INVALID_GL_OBJECT",
    "CL_INVALID_BUFFER_SIZE",
    "CL_INVALID_MIP_LEVEL",
    "CL_INVALID_GLOBAL_WORK_SIZE",
  };

  const int errorCount = sizeof(errorString) / sizeof(errorString[0]);

  const int index = -error;

  return (index >= 0 && index < errorCount) ? errorString[index] : "";
}

void check(cl_int r, string msg) {
  if(r != CL_SUCCESS)
  cerr << "error (" << msg << "): " << oclErrorString(r) << endl;
}
#endif

void getPSNR(unsigned char * inputImg, unsigned char * filteredImg, unsigned int height, unsigned int width, std::ostream &os) {
  Mat s1;
  Mat I1(height, width, CV_8U, inputImg, Mat::AUTO_STEP); // convert uc* to Mat
  Mat I2(height, width, CV_8U, filteredImg, Mat::AUTO_STEP); // convert uc* to Mat
  absdiff(I1, I2, s1);       // |I1 - I2|
  s1 = s1.mul(s1);           // |I1 - I2|^2

  Scalar s = sum(s1);         // sum elements per channel

  if (s.val[0] <= 1e-10) // for small values return zero
    os << " MAE/PSNR = 0! " << endl;
  else {
    double mse = s.val[0] / (double) (I1.total());
    os << " MAE = " << mse << " / ";
    double psnr = 10.0 * log10((255 * 255) / mse);
    os << " PSNR = " << psnr << " / ";
  }
}

void getMSSIM(unsigned char * inputImg, unsigned char * filteredImg, unsigned int height, unsigned int width, std::ostream &os) {
  const double C1 = 6.5025, C2 = 58.5225;
  /***************************** INITS **********************************/
  int d = CV_32F;

  Mat i1(height, width, CV_8U, inputImg, Mat::AUTO_STEP); // convert uc* to Mat
  Mat i2(height, width, CV_8U, filteredImg, Mat::AUTO_STEP); // convert uc* to Mat
  Mat I1;
  Mat I2;
  i1.convertTo(I1, d);           // cannot calculate on one byte large values
  i2.convertTo(I2, d);

  Mat I2_2 = I2.mul(I2);        // I2^2
  Mat I1_2 = I1.mul(I1);        // I1^2
  Mat I1_I2 = I1.mul(I2);        // I1 * I2

  /*************************** END INITS **********************************/

  Mat mu1, mu2;   // PRELIMINARY COMPUTING
  GaussianBlur(I1, mu1, Size(11, 11), 1.5);
  GaussianBlur(I2, mu2, Size(11, 11), 1.5);

  Mat mu1_2 = mu1.mul(mu1);
  Mat mu2_2 = mu2.mul(mu2);
  Mat mu1_mu2 = mu1.mul(mu2);

  Mat sigma1_2, sigma2_2, sigma12;

  GaussianBlur(I1_2, sigma1_2, Size(11, 11), 1.5);
  sigma1_2 -= mu1_2;

  GaussianBlur(I2_2, sigma2_2, Size(11, 11), 1.5);
  sigma2_2 -= mu2_2;

  GaussianBlur(I1_I2, sigma12, Size(11, 11), 1.5);
  sigma12 -= mu1_mu2;

  ///////////////////////////////// FORMULA ////////////////////////////////
  Mat t1, t2, t3;

  t1 = 2 * mu1_mu2 + C1;
  t2 = 2 * sigma12 + C2;
  t3 = t1.mul(t2);              // t3 = ((2*mu1_mu2 + C1).*(2*sigma12 + C2))

  t1 = mu1_2 + mu2_2 + C1;
  t2 = sigma1_2 + sigma2_2 + C2;
  t1 = t1.mul(t2);     // t1 =((mu1_2 + mu2_2 + C1).*(sigma1_2 + sigma2_2 + C2))

  Mat ssim_map;
  divide(t3, t1, ssim_map);      // ssim_map =  t3./t1;

  Scalar mssim = mean(ssim_map); // mssim = average of ssim map
  os << " mssim = " << mssim.val[0] << endl;
}

void printImageMetrics(unsigned char * originalImage, unsigned char * filteredImg, unsigned char * NoisyImg, unsigned int height,
    unsigned int width, std::ostream &os) {
  os << " Original x Noisy    : ";
  getPSNR(originalImage, NoisyImg, height, width, os);
  getMSSIM(originalImage, NoisyImg, height, width, os);
  os << " Original x Restored : ";
  getPSNR(originalImage, filteredImg, height, width, os);
  getMSSIM(originalImage, filteredImg, height, width, os);
}

#endif //_SPS_UTILS_HPP_
