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

#ifndef _SPD_POW_TABLE_HPP_
#define _SPD_POW_TABLE_HPP_
#include <cmath>

/*!
 * \class pow_table
 * \brief table for optimization s&p denoiser kernel
 *
 */
class pow_table {
public:
  pow_table(float exp) {
    table = (float *) malloc(256 * sizeof(float));
    for (int i = 0; i < 256; ++i)
      table[i] = pow((float) i, exp);
  }

  ~pow_table() {
    free(table);
  }

  float operator()(unsigned char a) {
    return table[a];
  }

private:
  float *table;
};
#endif
