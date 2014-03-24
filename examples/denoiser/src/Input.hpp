#ifndef INPUT_HPP_
#define INPUT_HPP_

#include <string>

/*!
 * \class Input
 *
 * \brief Reads frames from a video stream or bitmap (interface)
 *
 */
class Input {
public:

  /*!
   * \brief Fetches the next frame from the stream
   *
   * \return the array of pixels of the next frame
   */
  virtual unsigned char *nextFrame() = 0;

  unsigned int getHeight() {
    return height;
  }

  unsigned int getWidth() {
    return width;
  }

  /*!
   * \brief Returns the total number of frames of the stream
   *
   * \return the total number of frames of the stream
   */
  unsigned int getNFramesTotal() {
    return nFramesTotal;
  }

  virtual ~Input() {
  }

  double getFPS() {
    return fps;
  }

protected:
  unsigned int width, height;
  unsigned int nFramesTotal;
  double fps;
};
#endif
