#include <stdio.h>
#include <stdlib.h>
#include <utils.hpp>

#define N 1024

static unsigned long A[N][N];
static unsigned long B[N][N];
static unsigned long C[N][N];




int main() {
  
  /* init */
  for(int i=0;i<N;++i) 
	for(int j=0;j<N;++j) {
	  A[i][j] = i+j;
	  B[i][j] = i*j;
	  C[i][j] = 0;
	}
  
  ff::ffTime(ff::START_TIME);
  for(int i=0;i<N;++i) 
	for(int j=0;j<N;++j)
	  for(int k=0;k<N;++k)
		C[i][j] += A[i][k]*B[k][j];
  ff::ffTime(ff::STOP_TIME);
  std::cerr << "Computation time= " << ff::ffTime(ff::GET_TIME) << " (ms)\n";
  
#if 0
  for(int i=0;i<N;++i)  {
	  for(int j=0;j<N;++j)
	    printf(" %ld", C[i][j]);
	  
	  printf("\n");
    }
#endif
    return 0;
	
}


