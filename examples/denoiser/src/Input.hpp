/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */

/*
 *
 *  Authors:
 *    Maurizio Drocco
 *    Guilherme Peretti Pezzi 
 *  Contributors:  
 *    Marco Aldinucci
 *    Massimo Torquati
 *
 *  First version: February 2014
 */

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
