#ifndef _SPD_PARAMETERS_H_
#define _SPD_PARAMETERS_H_
//#include "definitions.hpp"
#include <string>
using namespace std;

#define MAX_CYCLES 2000
#define MAX_CYCLES_DEFAULT 200
#define MAX_WINDOW_SIZE (39*2+1)
#define ALFA_DEFAULT 1.3f
#define BETA_DEFAULT 5.0f
#define N_WORKERS_DEFAULT 2
#define FRACTION_SIZE_DEFAULT 1

typedef struct arguments {
  float alfa, beta;
  int w_max, max_cycles;
  bool fixed_cycles;
  int n_workers/*, fraction_size*/;
  string fname, out_fname;
  bool verbose;
  bool user_out_fname;
} arguments;

void print_help();
void get_arguments(char *argv[], int argc, arguments &args);
#endif
