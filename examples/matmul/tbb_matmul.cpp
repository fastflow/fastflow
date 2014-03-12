#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>

#include <ff/utils.hpp> // just for ffTime 


static long    N=0;      // matrix size
static double* A=NULL;   // this is pretty much equivalent to:  double * restrict A = NULL;
static double* B=NULL;   // this is pretty much equivalent to:  double * restrict B = NULL;
static double* C=NULL;   // this is pretty much equivalent to:  double * restrict C = NULL;

#if !defined(USE_LAMBDA)
class Mult {
public:
  void operator()(tbb::blocked_range<long> r) const {

    for (long i = r.begin(); i != r.end(); ++i) {
      for (long j = 0; j < N; ++j) {
	#pragma ivdep
	for (long k = 0; k < N; ++k) {
	  C[i*N+j] += A[i*N+k] * B[k*N+j];
	}
      }
    }
  }
};
#endif 

int main(int argc, char * argv[]) {
    if (argc<3) {
        std::cerr << "use: " << argv[0] << " nworkers size [K=5 check=false]\n";
        return -1;
    }

    int    K = 5;
    int    nworkers =atoi(argv[1]);
    N               =atol(argv[2]);
    assert(N>0);
    bool   check    =false;  
    if (argc>=4) K = atoi(argv[3]); 
    if (argc==5) {
	check=true;  // checks result
        if (K>1) printf("K is reset to 1 because of checking results\n");
	K=1;
    }
    
    A = (double*)malloc(N*N*sizeof(double));
    B = (double*)malloc(N*N*sizeof(double));
    C = (double*)malloc(N*N*sizeof(double));
    assert(A && B && C);

    tbb::task_scheduler_init init(nworkers);

    tbb::parallel_for(tbb::blocked_range<long>(0,N), 
		      [&] (const tbb::blocked_range<long>& r) {
			  for(long i=r.begin();i<r.end();++i) 
			      for(long j=0;j<N;++j) {
				  A[i*N+j] = (i+j)/(double)N;
				  B[i*N+j] = i*j*3.14;
				  C[i*N+j] = 0;
			      }
		      });

    ff::ffTime(ff::START_TIME);
    for(int q=0;q<K;++q) {
#if !defined(USE_LAMBDA)
	tbb::parallel_for(tbb::blocked_range<long>(0,N), Mult());
#else
	tbb::parallel_for(tbb::blocked_range<long>(0,N /*,
							 std::max( N/nworkers, (long)1)*/), 
			  [&] (const tbb::blocked_range<long>& r) {
			      for (long i = r.begin(); i != r.end(); ++i) {
				  for (long j = 0; j < N; ++j) {
                                      #pragma ivdep
				      for (long k = 0; k < N; ++k) {
					  C[i*N+j] += A[i*N+k] * B[k*N+j];
				      }
				  }
			      }			  
			  }
			  );
#endif
    } // q
    printf("%d Time = %g (ms) K=%d\n", nworkers,ff::ffTime(ff::STOP_TIME)/(1.0*K), K);
    
    if (check) {
        double R=0;
        for(long i=0;i<N;++i)
            for(long j=0;j<N;++j) {
                for(long k=0;k<N;++k)
                    R += A[i*N+k]*B[k*N+j];
		
                if (abs(C[i*N+j]-R)>1e-06) {
                    std::cerr << "Wrong result\n";                    
                    //printResults();
                    return -1;
                }
                R=0;
            }
        std::cout << "OK\n";
    }
    
    return 0;
}
