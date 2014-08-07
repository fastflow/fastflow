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
 

#ifndef RENDERTHREAD_H
#define RENDERTHREAD_H

#include <QMutex>
#include <QSize>
#include <QThread>
#include <QWaitCondition>

class QImage;

class RenderThread : public QThread
{
  Q_OBJECT

	friend class Worker; //here
	
    public:
  RenderThread(QObject *parent = 0);
  ~RenderThread();

  void render(double centerX, double centerY, double scaleFactor,
			  QSize resultSize);

  void stop_acc(); //ff shutdown

 signals:
  void renderedImage(const QImage &image, double scaleFactor);
  void stopped(); // ff shutdown

 protected:
  void run();

 private:
  uint rgbFromWaveLength(double wave);

  QMutex mutex;
  QWaitCondition condition;
  double centerX;
  double centerY;
  double scaleFactor;
  QSize resultSize;
  bool restart;
  bool abort;

  enum { ColormapSize = 512 };
  uint colormap[ColormapSize];

  bool cancel; // ff shutdown
};

#endif
