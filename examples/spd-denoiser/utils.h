#ifndef _SPS_UTILS_H_
#define _SPS_UTILS_H_

#include <string>
using namespace std;

#define _ABS(a)	   (((a) < 0) ? -(a) : (a))
#define _MAX(a, b) (((a) > (b)) ? (a) : (b))
#define _MIN(a, b) (((a) < (b)) ? (a) : (b))

//extract the filename from a path
string get_fname(string &path);

//time misuration
long int get_usec_from(long int s);

#ifdef CUDA
//get cuda environment and set the number of cores
void get_cuda_env(int &, bool);
#endif

#endif
