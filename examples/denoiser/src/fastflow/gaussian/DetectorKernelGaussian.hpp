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

#ifndef DETECTORGAUSSIAN_HPP_
#define DETECTORGAUSSIAN_HPP_

#include "opencv2/core/core.hpp"
#include <ocv/CVImageViewer.hpp>

using namespace cv;

/*!
 * \class DetectorGaussian
 *
 * \brief Check gaussian noise on a pixel
 *
 * This class detects gaussian noisy pixels
 * by applying a classical ACWMF.
 */
class DetectorKernelGaussian {
public:

  /*!
   * \brief Creates instance of DetectorGaussian
   *
   * It creates an instance of DetectorGaussian.
   *
   * \param height is the height of the frame
   * \param width is the width of the frame
   * \param trace_time_ tells if time should be measured
   */
  DetectorKernelGaussian(void *params_, unsigned int height_, unsigned int width_) :
    height(height_), width(width_),
      varianza(0), R(0), meanForVarianza(0), sommaForVarianza(0), outputValue(0), noisyVariance(0) {
    kernelSize = 3;
    K = 0;
    L = (int) ((kernelSize * kernelSize) - 1) / 2;
    TT = 4;
    windowSize = kernelSize * kernelSize;
    padding = kernelSize - 1;
    thresholdDetect = 0.0f;
    pixelVettForVariance = new float[windowSize];
    pixelVett = new float[windowSize];
    filtered = new Mat(height, width, CV_8U);
  }

  void init(unsigned char *noisyImage) {
    Mat orig(height, width, CV_8U, noisyImage, Mat::AUTO_STEP); // convert uc* to Mat
    IplImage iplNoisy = orig;
    ACWMF(&iplNoisy);
  }

  ~DetectorKernelGaussian() {
    delete[] pixelVettForVariance;
    delete[] pixelVett;
  }

  bool operator()(unsigned char *im, unsigned char center, unsigned int r, unsigned int c) {
    if (r < padding || c < padding || r >= height - padding || c >= width - padding)
      return false;
    unsigned char filt_value = filtered->at<uchar>((int) r, (int) c);
    return ((center + thresholdDetect) < filt_value || (center - thresholdDetect) > filt_value);
  }

private:
  unsigned int height, width;
  int kernelSize, K, L, TT, windowSize, padding;
  float R;
  float* pixelVett;
  float* pixelVettForVariance;
  float meanForVarianza;
  float sommaForVarianza;
  float varianza;
  float noisyVariance;
  float outputValue;
  float thresholdDetect;
  Mat * filtered; // result image from ACWMF, overwritten each time ACWMF is called

  void ACWMF(IplImage* im) {
    meanForVarianza = 0;
    sommaForVarianza = 0;
    varianza = 0;
    noisyVariance = 0;
    outputValue = 0;

    for (int j = padding; j < im->height - padding; j++) {
      for (int i = padding; i < im->width - padding; i++) {
        //CALCOLO DEL PARAMETRO R
        //Setto la dimensione del Kernel usato per il calcolo della varianza.
        //Per ogni intorno del pixel calcolo media e varianza.
        //I valori di media e varianza ottenuti li confronto con quelli usati per il rumore

        for (int s = -kernelSize / 2; s <= kernelSize / 2; s++) //Riempio la finestra con i valori del Kernel centrato nel pixel in questione.
          for (int t = -kernelSize / 2; t <= kernelSize / 2; t++) {
            *pixelVettForVariance = (float) cvGetReal2D(im, j + s, i + t);
            pixelVettForVariance++;
          }
        pixelVettForVariance = pixelVettForVariance - windowSize; //Setto l'indice della finestra al primo elemento dell'array

        //Calcolo della media: (sommatoria dei valori del kernel) / (il numero di elementi del kernel)
        //Eseguo la somma di tutti i valori del Kernel
        sommaForVarianza = 0;
        for (int count = 0; count < windowSize; count++) {
          sommaForVarianza = sommaForVarianza + *pixelVettForVariance;
          pixelVettForVariance++;
        }
        meanForVarianza = sommaForVarianza / windowSize;
        pixelVettForVariance = pixelVettForVariance - windowSize; //Setto l'indice della finestra al primo elemento dell'array

        sommaForVarianza = 0; //Calcolo della varianza: Eseguo la somma di tutti i valori del Kernel
        for (int count = 0; count < windowSize; count++) {
          sommaForVarianza = sommaForVarianza + pow((*pixelVettForVariance - meanForVarianza), 2);
          pixelVettForVariance++;
        }

        pixelVettForVariance = pixelVettForVariance - windowSize; //Setto l'indice della finestra al primo elemento dell'array
        varianza = sommaForVarianza / windowSize; //Calcolo il valore della varianza

        if (varianza > noisyVariance)
          R = (varianza - noisyVariance) / varianza;
        else
          R = 0;

        K = (int) ceil((float) (L - TT) * R);

        for (int s = -kernelSize / 2; s <= kernelSize / 2; s++) //Calcolo il valore della risposta all'impulso H
          for (int t = -kernelSize / 2; t <= kernelSize / 2; t++) {
            if (t == 0 && s == 0)
              *pixelVett = (float) ((2 * K) + 1) / (2 * L + 1);
            else
              *pixelVett = (float) (1 - (K / L)) / (2 * L + 1);
            pixelVett++;
          }
        pixelVett = pixelVett - windowSize; //Setto l'indice della finestra al primo elemento dell'array
        outputValue = 0; //Inizializzo il valore del pixel in uscita a 0

        for (int s = -kernelSize / 2; s <= kernelSize / 2; s++) { //Calcolo il valore del pixel in uscita dal filtro
          for (int t = -kernelSize / 2; t <= kernelSize / 2; t++) {
            outputValue = outputValue + (float) (cvGetReal2D(im, (j + s), (i + t)) * (*pixelVett));
            pixelVett++;
          }
        }
        pixelVett = pixelVett - windowSize; //Setto l'indice della finestra al primo elemento dell'array

        filtered->at<uchar>(j, i, 0) = outputValue + 0.5; //Assegno il valore del pixel al pixel dell'immagine
      }
    }
  }
};

#endif /* DETECTORGAUSSIAN_HPP_ */
