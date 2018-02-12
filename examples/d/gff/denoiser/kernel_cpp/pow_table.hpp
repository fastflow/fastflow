/* ***************************************************************************
 *
 *  This file is part of gam.
 *
 *  gam is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  gam is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *  See the GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with gam. If not, see <http://www.gnu.org/licenses/>.
 *
 ****************************************************************************
 */

#ifndef SPD_POW_TABLE_HPP_
#define SPD_POW_TABLE_HPP_
#include <cmath>

/*!
 * \class pow_table
 * \brief table for optimization s&p denoiser kernel
 *
 */
class pow_table {
public:
  pow_table(float exp) {
    for (int i = 0; i < 256; ++i)
      table[i] = pow((float) i, exp);
  }

  float operator()(int a) {
    return table[a];
  }

private:
  float table[256];
};
#endif
