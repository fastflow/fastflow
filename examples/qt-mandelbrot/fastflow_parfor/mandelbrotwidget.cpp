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
 

#include <QtGui>

#include <math.h>
#include <iostream> 
#include "mandelbrotwidget.h"

const double DefaultCenterX = -0.637011f;
const double DefaultCenterY = -0.0395159f;
const double DefaultScale = 0.00403897f;

const double ZoomInFactor = 0.8f;
const double ZoomOutFactor = 1 / ZoomInFactor;
const int ScrollStep = 20;

MandelbrotWidget::MandelbrotWidget(QWidget *parent)
  : QMainWindow(parent)
{

  
  createActions();
  createMenus();

  centerX = DefaultCenterX;
  centerY = DefaultCenterY;
  pixmapScale = DefaultScale;
  curScale = DefaultScale;
  
  qRegisterMetaType<QImage>("QImage");
  connect(&thread, SIGNAL(renderedImage(const QImage &, double)),
		  this, SLOT(updatePixmap(const QImage &, double)));
  //connect(&thread, SIGNAL(stopped()),this, SLOT(shutdown())); // ff
  setWindowTitle(tr("Mandelbrot"));
  setCursor(Qt::CrossCursor);
  resize(550, 400);
}

void MandelbrotWidget::paintEvent(QPaintEvent * /* event */)
{
  QPainter painter(this);
  painter.fillRect(rect(), Qt::black);

  if (pixmap.isNull()) {
	painter.setPen(Qt::white);
	painter.drawText(rect(), Qt::AlignCenter,
					 tr("Rendering initial image, please wait..."));
	return;
  }
  
  if (curScale == pixmapScale) {
	painter.drawPixmap(pixmapOffset, pixmap);
  } else {
	double scaleFactor = pixmapScale / curScale;
	int newWidth = int(pixmap.width() * scaleFactor);
	int newHeight = int(pixmap.height() * scaleFactor);
	int newX = pixmapOffset.x() + (pixmap.width() - newWidth) / 2;
	int newY = pixmapOffset.y() + (pixmap.height() - newHeight) / 2;
	
	painter.save();
	painter.translate(newX, newY);
	painter.scale(scaleFactor, scaleFactor);
	painter.drawPixmap(0, 0, pixmap);
	painter.restore();
  }
  
  QString text = tr("Use mouse wheel to zoom. "
					"Press and hold left mouse button to scroll.");
  QFontMetrics metrics = painter.fontMetrics();
  int textWidth = metrics.width(text);
  
  painter.setPen(Qt::NoPen);
  painter.setBrush(QColor(0, 0, 0, 127));
  painter.drawRect((width() - textWidth) / 2 - 5, 0, textWidth + 10,
				   metrics.lineSpacing() + 5);
  painter.setPen(Qt::white);
  painter.drawText((width() - textWidth) / 2,
				   metrics.leading() + metrics.ascent(), text);
}

void MandelbrotWidget::resizeEvent(QResizeEvent * /* event */)
{
  thread.render(centerX, centerY, curScale, size());
}

void MandelbrotWidget::keyPressEvent(QKeyEvent *event)
{
  switch (event->key()) {
  case Qt::Key_Plus:
	zoom(ZoomInFactor);
	break;
  case Qt::Key_Minus:
	zoom(ZoomOutFactor);
	break;
  case Qt::Key_Left:
	scroll(-ScrollStep, 0);
	break;
  case Qt::Key_Right:
	scroll(+ScrollStep, 0);
	break;
  case Qt::Key_Down:
	scroll(0, -ScrollStep);
	break;
  case Qt::Key_Up:
	scroll(0, +ScrollStep);
	break;
  default:
	QWidget::keyPressEvent(event);
  }
}

void MandelbrotWidget::wheelEvent(QWheelEvent *event)
{
  int numDegrees = event->delta() / 8;
  double numSteps = numDegrees / 15.0f;
  zoom(pow(ZoomInFactor, numSteps));
}

void MandelbrotWidget::mousePressEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton)
	lastDragPos = event->pos();
}

void MandelbrotWidget::mouseMoveEvent(QMouseEvent *event)
{
  if (event->buttons() & Qt::LeftButton) {
	pixmapOffset += event->pos() - lastDragPos;
	lastDragPos = event->pos();
	update();
  }
}

void MandelbrotWidget::mouseReleaseEvent(QMouseEvent *event)
{
  if (event->button() == Qt::LeftButton) {
	pixmapOffset += event->pos() - lastDragPos;
	lastDragPos = QPoint();
	
	int deltaX = (width() - pixmap.width()) / 2 - pixmapOffset.x();
	int deltaY = (height() - pixmap.height()) / 2 - pixmapOffset.y();
	scroll(deltaX, deltaY);
  }
}

void MandelbrotWidget::updatePixmap(const QImage &image, double scaleFactor)
{
  if (!lastDragPos.isNull())
	return;
  
  pixmap = QPixmap::fromImage(image);
  pixmapOffset = QPoint();
  lastDragPos = QPoint();
  pixmapScale = scaleFactor;
  update();
}

void MandelbrotWidget::zoom(double zoomFactor)
{
  curScale *= zoomFactor;
  update();
  thread.render(centerX, centerY, curScale, size());
}

void MandelbrotWidget::scroll(int deltaX, int deltaY)
{
  centerX += deltaX * curScale;
  centerY += deltaY * curScale;
  update();
  thread.render(centerX, centerY, curScale, size());
}


// for performance testing purpose only, not in the original code. 

void MandelbrotWidget::testact0() {
  // Default
  std::cout << "[Widget] Default test\n";
  centerX = -0.637011f;
  centerY = -0.0395159f;
  curScale = 0.00403897f;
  thread.render(centerX, centerY, curScale, size());
}

void MandelbrotWidget::testact1() {
  // Wreath
  std::cout << "[Widget] Wreath test\n";
  centerX = -0.756423894629558;
  centerY = 0.064179410844781;
  curScale = -0.00000000003300000;
  thread.render(centerX, centerY, curScale, size());
}

void MandelbrotWidget::testact2() {
  // Broccoli 
  std::cout << "[Widget] Broccoli test\n";
  centerX = -1.746544044825862;
  centerY = -0.000016605493831;
  curScale = 0.00000000247437599;
  thread.render(centerX, centerY, curScale, size()); 
}

void MandelbrotWidget::testact3() {
  // Double Helix
  std::cout << "[Widget] Double Helix test\n";
  centerX = -1.941558464320810;
  centerY = 0.000112771806258;
  curScale = 0.00000000382674917;
  thread.render(centerX, centerY, curScale, size()); 
}

void MandelbrotWidget::close_acc() {
  // Ask shutdown to renderthread
  //thread.stop_acc(); 
}

//receive signal from renderthread: accelerator is off 
void MandelbrotWidget::shutdown() { 
  // now exit
  close();
}

 void MandelbrotWidget::createActions()
 {
   ZeroAct = new QAction(tr("Default"), this);
   connect(ZeroAct, SIGNAL(triggered()), this, SLOT(testact0()));
   
   OneAct = new QAction(tr("Wreath"), this);
   connect(OneAct, SIGNAL(triggered()), this, SLOT(testact1()));
   
   TwoAct = new QAction(tr("Broccoli"), this);
   connect(TwoAct, SIGNAL(triggered()), this, SLOT(testact2()));
   
   ThreeAct = new QAction(tr("Double Helix"), this);
   connect(ThreeAct, SIGNAL(triggered()), this, SLOT(testact3()));

   // Exit procedure should be re-defined to properly shutdown the accelerator
   exitAct = new QAction(tr("&Quit"), this);
   // exitAct->setShortcuts(QKeySequence::Quit);            // requires QT 4.6
   // exitAct->setStatusTip(tr("Exit the application"));
   connect(exitAct, SIGNAL(triggered()), this, SLOT(close_acc()));
  
 }

void MandelbrotWidget::createMenus()
 {
     testMenu = new QMenu(tr("&Test"), this);
     testMenu->addAction(OneAct);
     testMenu->addAction(TwoAct);
     testMenu->addAction(ThreeAct);
	 testMenu->addSeparator();
	 testMenu->addAction(ZeroAct);
	 testMenu->addSeparator();
	 testMenu->addAction(exitAct); // Quit -> accelerator shutdown
     menuBar()->addMenu(testMenu);
 }


