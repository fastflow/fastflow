#ifndef _SPD_CAPTURE_UTILS_HPP_
#define _SPD_CAPTURE_UTILS_HPP_
int adjust(CvCapture *capture, char key, int noise, int c_size[4][2], int *c_size_i, unsigned int *height, unsigned int *width) {
  if (key=='+' && noise < 85) {
    noise +=10;
    cout << "Noise " << noise << "%\n";
  }
  if (key=='-' && noise > 15) {
    noise -=10;
    cout << "Noise " << noise << "%\n";
  }
  if ((key=='l') && (*c_size_i < 4)) {
    (*c_size_i)++;
    *width = c_size[*c_size_i][0];
    *height = c_size[*c_size_i][1];
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH, (int)*width);
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT, (int)*height);
  }
  if ((key=='s') && (*c_size_i > 0)) {
    (*c_size_i)--;
    *width = c_size[*c_size_i][0];
    *height = c_size[*c_size_i][1];
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_WIDTH, (int)*width);
    cvSetCaptureProperty(capture,CV_CAP_PROP_FRAME_HEIGHT, (int)*height);
  }
  if (noise>85) noise=85;
  if (noise<0) noise=0;
  return noise;
}
#endif
