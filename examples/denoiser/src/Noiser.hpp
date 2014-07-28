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
