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
