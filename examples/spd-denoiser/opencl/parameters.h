#ifndef _SPD_PARAMETERS_H_
#define _SPD_PARAMETERS_H_
#include <string>
using namespace std;

#if defined(FROMFILE) | defined(FROMCAMERA)
#define OUTFILE_EXT "avi"
#endif
#ifdef BITMAP
#define OUTFILE_EXT "bmp"
#define WAIT_FOR 3
#endif
#ifdef BITMAPCOLOR
#define OUTFILE_EXT "bmp"
#define WAIT_FOR 3
#endif

#define MAX_CYCLES 2000
#define MAX_CYCLES_DEFAULT 200
#define MAX_WINDOW_SIZE (39*2+1)
#define ALFA_DEFAULT 1.3f
#define BETA_DEFAULT 5.0f

typedef struct arguments {
  float alfa, beta;
  int w_max, max_cycles, noise;
  bool fixed_cycles;
  string fname, out_fname;
  bool verbose, show_enabled, user_out_fname, add_noise;
} arguments;

void print_help();
void get_arguments(char *argv[], int argc, arguments &args);
#endif
