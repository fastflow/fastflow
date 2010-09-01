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
 

#ifndef MANDELBROTWIDGET_H
#define MANDELBROTWIDGET_H

#include <QMainWindow>
#include <QPixmap>
#include <QWidget>

#include "renderthread.h"

class QMenu;
class QAction;
class QLabel;

class MandelbrotWidget : public QMainWindow
{
  Q_OBJECT
	
    public:
  MandelbrotWidget(QWidget *parent = 0);
  
 protected:
  void paintEvent(QPaintEvent *event);
  void resizeEvent(QResizeEvent *event);
  void keyPressEvent(QKeyEvent *event);
  void wheelEvent(QWheelEvent *event);
  void mousePressEvent(QMouseEvent *event);
  void mouseMoveEvent(QMouseEvent *event);
  void mouseReleaseEvent(QMouseEvent *event);
  
  private slots:
  void updatePixmap(const QImage &image, double scaleFactor);
  void testact0();
  void testact1();
  void testact2();
  void testact3();
  void close_acc(); // ff
  void shutdown(); // ff
  
 private:
  void zoom(double zoomFactor);
  void scroll(int deltaX, int deltaY);
  
  RenderThread thread;
  QPixmap pixmap;
  QPoint pixmapOffset;
  QPoint lastDragPos;
  double centerX;
  double centerY;
  double pixmapScale;
  double curScale;

  void createMenus();
  void createActions();
  QAction *exitAct; //
  QAction *ZeroAct;
  QAction *OneAct;
  QAction *TwoAct;
  QAction *ThreeAct;
  QMenu *testMenu;
};

#endif
