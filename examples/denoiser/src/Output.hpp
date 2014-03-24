#ifndef OUTPUT_HPP_
#define OUTPUT_HPP_

#include <string>

/*!
 * \class Output
 *
 * \brief Writes frames to a video stream or to a bitmap (interface)
 *
 * This class writes frames to a video stream.
 */
class Output {
public:
  /*!
   * \brief Initializes the output engine
   *
   */
  Output(int height_, int width_) :
      width(width_), height(height_) {
  }
  ;

  /*!
   * \brief Writes a frame to the stream
   *
   * \param frame is the frame to be written
   */
  virtual void writeFrame(unsigned char *frame) = 0;

  virtual ~Output() {
  }

protected:
  unsigned int width, height;
};

#endif
