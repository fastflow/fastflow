/****************************************************************************
 **
 ** Copyright (C) 2004-2006 Trolltech ASA. All rights reserved.
 **
 ** This file is part of the documentation of the Qt Toolkit.
 **
 ** This file may be used under the terms of the GNU General Public
 ** License version 2.0 as published by the Free Software Foundation
 ** and appearing in the file LICENSE.GPL included in the packaging of
 ** this file.  Please review the following information to ensure GNU
 ** General Public Licensing requirements will be met:
 ** http://www.trolltech.com/products/qt/opensource.html
 **
 ** If you are unsure which license is appropriate for your use, please
 ** review the following information:
 ** http://www.trolltech.com/products/qt/licensing.html or contact the
 ** sales department at sales@trolltech.com.
 **
 ** This file is provided AS IS with NO WARRANTY OF ANY KIND, INCLUDING THE
 ** WARRANTY OF DESIGN, MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE.
 **
 ****************************************************************************/

// Author: Marco Aldinucci - aldinuc@di.unito.it
// Note: Porting to FF or the QT Mandelbrot example
// Date: Dec 2009

extern int nworkers;
 
#include <QtGui>

#include <math.h>

#include "renderthread.h"

// Additional headers ---
#include <vector>
#include <iostream>
#include <ff/farm.hpp>
#include <ff/node.hpp>
#include <ff/utils.hpp>
#include <ff/allocator.hpp>

using namespace ff;

/* Uncomment the following define if you want to test the 
 * FastFlow memory allocator 
 */
//#define USE_FFA

#if defined(USE_FFA)
static ff_allocator * ffalloc = NULL;
#endif



// ----------------------

// Additional classes ---

typedef struct accelerator_closure {
  // RenderThread * rt; // to access all const variables
  double ay;
  double halfWidth;
  int Limit;
  int MaxIterations;
  uint * scanLine;
  bool * allBlack_p;
} accelerator_closure_t;


class Worker: public ff_node {
public:
  Worker(RenderThread * const caller):rt(caller){}

#if defined(USE_FFA)
  int svc_init() {
      if (ffalloc->register4free()<0) {
	  error("Worker, register4free fails\n");
	  return -1;
      }
      return 0;
  }
#endif

  void * svc(void * task) {
	accelerator_closure_t * t = (accelerator_closure_t *)task;
	//std::cout << "[Worker] received task " << t << "\n";
	int index = 0; // here - added
  
	// -- copy-pasted from original code
	if (rt->restart)
	  return task;
	if (rt->abort)
	  return task;
	for (int x = -t->halfWidth; x < t->halfWidth; ++x) {
	  double ax = rt->centerX + (x * rt->scaleFactor); // alpha-renaming
	  double a1 = ax;
	  double b1 = t->ay; // alpha-renaming
	  int numIterations = 0;
	  do {
		++numIterations;
		double a2 = (a1 * a1) - (b1 * b1) + ax;
		double b2 = (2 * a1 * b1) + t->ay; // alpha-renaming
		if ((a2 * a2) + (b2 * b2) > t->Limit) // alpha-renaming
		  break;
		  
		++numIterations;
		a1 = (a2 * a2) - (b2 * b2) + ax;
		b1 = (2 * a2 * b2) + t->ay; // alpha-renaming
		if ((a1 * a1) + (b1 * b1) > t->Limit) // alpha-renaming
		  break;
	  } while (numIterations < t->MaxIterations);
		
	  if (numIterations < t->MaxIterations) {
		t->scanLine[index++] = rt->colormap[numIterations % rt->ColormapSize]; // alpha-renaming
		*(t->allBlack_p) = false; // alpha-renaming
	  } else {
		t->scanLine[index++] = qRgb(0, 0, 0); // alpha-renaming
	  }
	}
	// -----
#if defined(USE_FFA)
	ffalloc->free(t);
#else
	delete t;
#endif
	return GO_ON;
  }
private:
  RenderThread *  rt;
};

// --------------

RenderThread::RenderThread(QObject *parent)
  : QThread(parent)
{
  restart = false;
  abort = false;
  cancel = false; // ff
  
  for (int i = 0; i < ColormapSize; ++i)
	colormap[i] = rgbFromWaveLength(380.0 + (i * 400.0 / ColormapSize));

#if defined(USE_FFA)
  // register FF allocator
  std::cout << "Allocator registering \n";
  ffalloc = new ff_allocator;
  if (ffalloc && ffalloc->init()<0) {
	error("cannot build ff_allocator\n");
	return;
  }
  if (ffalloc->registerAllocator()<0) {
	error("Emitter, registerAllocator fails\n");
	return;
  }
  std::cout << "Allocator registered \n";
  // ---------  
#endif

}

RenderThread::~RenderThread()
{
  mutex.lock();
  abort = true;
  condition.wakeOne();
  mutex.unlock();

  wait();
}

void RenderThread::render(double centerX, double centerY, double scaleFactor,
						  QSize resultSize)
{
  QMutexLocker locker(&mutex);

  this->centerX = centerX;
  this->centerY = centerY;
  this->scaleFactor = scaleFactor;
  this->resultSize = resultSize;

  if (!isRunning()) {
	start(LowPriority);
  } else {
	restart = true;
	condition.wakeOne();
  }
}

// stop the accelerator - wake renderthread up in the case it if sleeping
void RenderThread::stop_acc() {
  cancel = true;
  mutex.lock();
  condition.wakeOne();
  mutex.unlock();
}

void RenderThread::run()
{
  struct timeval t0,t1;

   // FF-additional code - Create and start the accelerator
  //int nworkers = sysconf(_SC_NPROCESSORS_ONLN);
  ff_farm<> farm(true /* accelerator set */, 256*nworkers);
  std::vector<ff_node *> w;
  for(int i=0;i<nworkers;++i) w.push_back(new Worker(this));
  farm.add_workers(w);
  //Collector C;
  //farm.add_collector(&C);   

  forever {
	std::cout << "[Renderthread] receiving parameters from main QT thread ... ";
	mutex.lock();
	QSize resultSize = this->resultSize;
	double scaleFactor = this->scaleFactor;
	//double centerX = this->centerX;
	double centerY = this->centerY;
	mutex.unlock();
	std::cout << "[Renderthread] done.\n";
	
	
	int halfWidth = resultSize.width() / 2;
	int halfHeight = resultSize.height() / 2;
	QImage image(resultSize, QImage::Format_RGB32);
	
	const int NumPasses = 8;
	int pass = 0;
	while (pass < NumPasses) {

	  farm.run_then_freeze();

	  gettimeofday(&t0,NULL);
	 
	  //farm.run_then_freeze();
	  //std::cout << "[Renderthread] Farm accelerator started on pass " << pass << "\n";
	  // -------
	  
	  const int MaxIterations = (1 << (2 * pass + 6)) + 32;
	  const int Limit = 4;
	  bool allBlack = true;
	  
	  for (int y = -halfHeight; y < halfHeight; ++y) {
		if (restart)
		  break;
		if (abort)
		  return;
		
		uint *scanLine =
		  reinterpret_cast<uint *>(image.scanLine(y + halfHeight));
		double ay = centerY + (y * scaleFactor);
		// --- parameter passing of cycle variables
#if defined(USE_FFA)
		accelerator_closure_t * ac = (accelerator_closure_t *)ffalloc->malloc(sizeof(accelerator_closure_t));
#else
		accelerator_closure_t * ac = new accelerator_closure_t();
#endif
		ac->ay = ay;
		ac->halfWidth = halfWidth;
		ac->Limit = Limit;
		ac->MaxIterations = MaxIterations;
		ac->scanLine = scanLine;
		ac->allBlack_p =  &allBlack;
		
		// check shutdown
		if (cancel) {
		  //farm.offload((void *)FF_EOS);
		  break;
		}

		farm.offload(ac);
		// --------------
		// parallelized code, see worker
		/*
		  for (int x = -halfWidth; x < halfWidth; ++x) {
		  double ax = centerX + (x * scaleFactor);
		  double a1 = ax;
		  double b1 = ay;
		  int numIterations = 0;

		  do {
		  ++numIterations;
		  double a2 = (a1 * a1) - (b1 * b1) + ax;
		  double b2 = (2 * a1 * b1) + ay;
		  if ((a2 * a2) + (b2 * b2) > Limit)
		  break;
		  
		  ++numIterations;
		  a1 = (a2 * a2) - (b2 * b2) + ax;
		  b1 = (2 * a2 * b2) + ay;
		  if ((a1 * a1) + (b1 * b1) > Limit)
		  break;
		  } while (numIterations < MaxIterations);
		  
		  if (numIterations < MaxIterations) {
		  *scanLine++ = colormap[numIterations % ColormapSize];
		  allBlack = false;
		  } else {
		  *scanLine++ = qRgb(0, 0, 0);
		  }
		  }
		*/
	  }
	  // FF-additional code - Stop the accelerator
	  void * eos = (void *)FF_EOS;
	  farm.offload(eos);
	  //std::cout << "[Renderthread] waiting ...\n";
	  farm.wait_freezing(); 
	  gettimeofday(&t1,NULL);
	  std::cout << "[Renderthread] pass " << pass << " done [" << diffmsec(t1,t0) << "ms]\n"; 
	  // ---------------
	  if (allBlack && pass == 0) {
		pass = 4;
	  } else {
		if (!restart)
		  emit renderedImage(image, scaleFactor);
		++pass;
	  }
	}
	
	// FF-additional code - Stop the accelerator
	//std::cout << "[Renderthread] EOS arrived\n";
	//void * eos = (void *)FF_EOS;
	//farm.offload(eos);
	farm.wait();
	if (cancel) {
	  std::cout << "[Renderthread] Accelerator is now off \n";
	  emit stopped();
	  return;
	}
	// ---------------
	
	std::cout << "[Renderthread] work completed, going on wait state\n";
	mutex.lock();
	if (!restart)
	  condition.wait(&mutex);
	restart = false;
	mutex.unlock();
  }
}

uint RenderThread::rgbFromWaveLength(double wave)
{
  double r = 0.0;
  double g = 0.0;
  double b = 0.0;

  if (wave >= 380.0 && wave <= 440.0) {
	r = -1.0 * (wave - 440.0) / (440.0 - 380.0);
	b = 1.0;
  } else if (wave >= 440.0 && wave <= 490.0) {
	g = (wave - 440.0) / (490.0 - 440.0);
	b = 1.0;
  } else if (wave >= 490.0 && wave <= 510.0) {
	g = 1.0;
	b = -1.0 * (wave - 510.0) / (510.0 - 490.0);
  } else if (wave >= 510.0 && wave <= 580.0) {
	r = (wave - 510.0) / (580.0 - 510.0);
	g = 1.0;
  } else if (wave >= 580.0 && wave <= 645.0) {
	r = 1.0;
	g = -1.0 * (wave - 645.0) / (645.0 - 580.0);
  } else if (wave >= 645.0 && wave <= 780.0) {
	r = 1.0;
  }

  double s = 1.0;
  if (wave > 700.0)
	s = 0.3 + 0.7 * (780.0 - wave) / (780.0 - 700.0);
  else if (wave <  420.0)
	s = 0.3 + 0.7 * (wave - 380.0) / (420.0 - 380.0);

  r = pow(r * s, 0.8);
  g = pow(g * s, 0.8);
  b = pow(b * s, 0.8);
  return qRgb(int(r * 255), int(g * 255), int(b * 255));
}
