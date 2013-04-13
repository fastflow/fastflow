#ifndef __NOISER_HPP__
#define __NOISER_HPP__

// Salt-and-Pepper_Noise_Removal_GRAYSCALE
// Filtro a due passi a più cicli per la rimozione del disturbo 'Salt & Pepper' da immagini bitmap a 8 bit in scala di grigi.
#include <opencv/cv.h>
#include <opencv/cvaux.h>
#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include <vector>


using namespace std;

void addSaltandPepperNoiseDEPTH_8U (IplImage* img, double percent)
{
  //cout << "   Adding " << percent << "% salt and pepper noise to the image...";
  int count = 0;
  //int width = img->width;
  //int height = img->height;

  double pr = 1.0-percent/100.0;
  for (int i = 0; i < img->height; ++i)
    for (int j = 0; j < img->width; ++j)
      {
	// Generate random number between -1.0 and +1.0
    double rr = (double) rand( ); 
    double random = 2.0*(rr - (RAND_MAX) / 2.0) / (RAND_MAX);
	if (random > pr)
	  {
		((uchar *)(img->imageData + i*img->widthStep))[j]=255;
	    //bmp.set(j,i, (grayscale)255);	// Salt noise
	    ++count;
	  }
	else if (random < -pr)
	  {
		((uchar *)(img->imageData + i*img->widthStep))[j]=0;
	    //bmp.set(j,i, (grayscale)0);    // Pepper noise
	    ++count;
	  }
      }
  double actualpercent = 100.0*count/(img->width*img->height);

  //cout << "done! (" << actualpercent << "%)" << endl;
}
/*
void addSaltandPepperNoise (Bitmap<> &bmp, double percent)
{
  cout << "   Adding " << percent << "% salt and pepper noise to the image...";
  srand(1);
  int count = 0;

  bmp_size_t width = bmp.width();
  bmp_size_t height = bmp.height();

  double pr = 1.0-percent/100.0;
  for (int i = 0; i < width; ++i)
    for (int j = 0; j < height; ++j)
      {
	// Generate random number between -1.0 and +1.0
	double random = 2.0*(rand()-RAND_MAX/2.0)/RAND_MAX;
	if (random > pr)
	  {
	    bmp.set(j,i, (grayscale)255);	// Salt noise
	    ++count;
	  }
	else if (random < -pr)
	  {
	    bmp.set(j,i, (grayscale)0);    // Pepper noise
	    ++count;
	  }
      }
  double actualpercent = 100.0*count/(width*height);

  cout << "done! (" << actualpercent << "%)" << endl;
}

*/

//***************************************************************************
/*
void GrayImage::addGaussianNoise (double mean, double variance)
{
cout << "  Adding Gaussian noise with mean = " << mean << " and variance = "
<< variance << " to the image...";
cout.flush();
srand(1);
#define unit_random() (1.0*rand()/RAND_MAX)
#define TWO_PI 6.28318530717958647688

double temp, u1, u2;
int ix, iy;
int num = width*height;
int tempint;

u1=0.0;
for(int i = 0; i < num/2; i++)
{
while (u1 == 0.0) u1 = unit_random();
u2 = unit_random();
temp = sqrt(-2.0*variance*log(u1));
ix = 2*i/width;
iy = 2*i%width;
tempint = p[ix][iy] + (int) (temp * cos(TWO_PI*u2) + mean);
if (tempint > 255)
p[ix][iy] = 255;
else if (tempint < 0)
p[ix][iy] = 0;
else
p[ix][iy] = (unsigned char) tempint;
ix = (2*i+1)/width;
iy = (2*i+1)%width;
tempint = p[ix][iy] + (int) (temp * sin(TWO_PI*u2) + mean);
if (tempint > 255)
p[ix][iy] = 255;
else if (tempint < 0)
p[ix][iy] = 0;
else
p[ix][iy] = (unsigned char) tempint;
u1 = 0.0;
}

u1=0.0;
if(num & 1) // If num is odd
{
while (u1 == 0.0) u1 = unit_random();
u2 = unit_random();
temp = sqrt(-2.0*variance*log(u1));
ix = (num-1)/width;
	 iy = (num-1)%width;
	 tempint = p[ix][iy] + (int) (temp * cos(TWO_PI*u2) + mean);
	 if (tempint > 255)
		p[ix][iy] = 255;
	 else if (tempint < 0)
		p[ix][iy] = 0;
	 else
		p[ix][iy] = (unsigned char) tempint;
	 u1 = 0.0;
	}

cout << "done!" << endl;
}
*/
//***************************************************************************
#endif


