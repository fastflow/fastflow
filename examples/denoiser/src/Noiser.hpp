#ifndef NOISER_HPP_
#define NOISER_HPP_

/*!
 * \class Noiser
 *
 * \brief Adds noise to a frame (interface)
 *
 * This class adds noise to a frame.
 */
class Noiser {
public:
  Noiser(unsigned int height_, unsigned int width_) :
      height(height_), width(width_) {
  }

  /*!
   * \brief Adds noise to a frame
   *
   * It adds noise to a frame.
   *
   * \param im is the input image
   */
  virtual void addNoise(unsigned char *im) = 0;

  virtual ~Noiser() {}

protected:
  unsigned int height, width;

};
#endif
