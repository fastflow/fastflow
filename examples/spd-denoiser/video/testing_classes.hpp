

class Capture: public ff_node {
public:
  //Capture() {}
  Capture(CvCapture *capture): capture(capture) {
	if (capture == NULL) {
	  std::cerr << "Capture device is NULL\n";
		exit(-1);
	}	  
  }

  void * svc(void * task) {
	IplImage * frame;
	if (!task) {
	  key = cvWaitKey(100);
	  if (key != 'q' && key != 'Q') {
		imagep = (IplImage **) malloc(sizeof(void *));
		while (true) {
		  frame = cvQueryFrame(capture);
		  cerr << "Capture - frame " << frame << "\n";
		  //*imagep = cvCloneImage(frame);
		  cvShowImage( "Window", frame);
		  key = cvWaitKey(100);
		}
		cerr << "Capture - frame " << frame << "\n";
		//*imagep=cvRetrieveFrame(capture);
		//cvShowImage( "Window", *imagep);
		//key=cvWaitKey(20);
		return imagep;
	  } else
		return NULL;
	}	
  }

private:
  IplImage **imagep;
  CvCapture *capture;
  char key;
};

class Show: public ff_node {
public:

  Show(CvCapture *capture): capture(capture) {}

  /*
  int svc_init () {
	cvNamedWindow("Window", CV_WINDOW_AUTOSIZE);
  }

  void  svc_end() {
	cvDestroyWindow("Window");
  }
  */

  void * svc(void * task) {
	IplImage **imagep = (IplImage **) task;
	
	if (imagep) {
	  cerr << "Show - task " << task << " image " << *imagep << " \n";
	  //cvShowImage( "Window", *imagep);
	  //cvReleaseImage(imagep);
	  return task;
	}
  }
private:
  CvCapture *capture;
};

/// End basic non-working pipeline

//Begin - test farm accelerator
class Filter: public ff_node {
public:

  Filter() {}

  void * svc(void * task) {
	IplImage *imagep = (IplImage *) task;
	IplImage **gray = NULL;
	if (imagep) {
	  cerr << " Filter - task " << task << " image " << imagep << " \n";
	  gray = (IplImage **) malloc(sizeof(void *));
	  *gray = cvCreateImage( cvSize(imagep->width, imagep->height), IPL_DEPTH_8U, 1);
	  cvCvtColor(imagep, *gray, CV_BGR2GRAY);
	  cvReleaseImage(&imagep);
	  free(imagep);
	  //sleep(1);
	  return *gray;
	}
  }
  //private:
  //CvCapture *capture;
};

class Collector: public ff_node {
public:
    void * svc(void * task) {        
        return task;  
    }
};
