#include <opencv/highgui.h>
#include <opencv/cv.h>
#include <ff/pipeline.hpp>

typedef struct noisy_img_task {
  Bitmap<> *bmp;
  vector<noisy_t> *the_noisy_set;
  typedef struct noisy_img_task {
  Bitmap<> *bmp;
  noisy_set *the_noisy_set;
  noisy_img_task(Bitmap<> *bmp): bmp(bmp) {};
} noisy_img_task_t;

int main(int argc, char *argv[]) {
  CvCapture *capture;
  IplImage * frame,clean_frame;
  char key;
  vector<noisy_t> noisy;

  ff_pipeline pipe(true);
  pipe.add_stage(new myDetect());      // detect noisy pixels
  pipe.add_stage(new myDenoise());     // denoise the frame
  pipe.run();

  cvNamedWindow("Restored", CV_WINDOW_AUTOSIZE);
  capture = cvCreateCameraCapture(CV_CAP_ANY);
  //capture = cvCreateFileCapture("/path/to/your/video/test.avi");
  while(true) {
	frame = cvQueryFrame(capture);        // get a frame from device
	noisy_img_task * nt = new noisy_img_task(frame); //privatisation
	pipe.offload(nt);                  // offload detect and denoise
	do {
	  while (pipe.load_result_nb(&result)) {
		--inflight;
		noisy_img_task_t * rt = (noisy_img_task_t *) result;
		delete rt;
		cvShowImage( "Restored", result->bmp); 
		key = cvWaitKey(100);
	  } 
	} while (inflight>5 /* MAXINFLIGHT */);
	
  }
  pipe.offload((void *)FF_EOS);      // cleaning up
  pipe.wait();                      
  cvReleaseCapture(&capture);
  cvDestroyWindow("Restored");
}
