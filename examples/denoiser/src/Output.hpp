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
