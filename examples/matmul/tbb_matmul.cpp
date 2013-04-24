#include <tbb/parallel_for.h>
#include <tbb/blocked_range.h>
#include <tbb/task_scheduler_init.h>

#include <ff/utils.hpp> // just for ffTime 


static long    N=0;      // matrix size
static double* A=NULL;
static double* B=NULL;
static double* C=NULL;

class Mult {
public:
  void operator()(tbb::blocked_range<long> r) const {

    for (long i = r.begin(); i != r.end(); ++i) {
      for (long j = 0; j < N; ++j) {
	for (long k = 0; k < N; ++k) {
	  C[i*N+j] += A[i*N+k] * B[k*N+j];
	}
      }
    }
  }
};

int main(int argc, char * argv[]) {
    if (argc<3) {
        std::cerr << "use: " << argv[0] << " nworkers size [check]\n";
        return -1;
    }

    int    nworkers =atoi(argv[1]);
    N               =atol(argv[2]);
    assert(N>0);
    bool   check    =false;  
    if (argc==4) check=true;  // checks result
    
    A = (double*)malloc(N*N*sizeof(double));
    B = (double*)malloc(N*N*sizeof(double));
    C = (double*)malloc(N*N*sizeof(double));
    assert(A && B && C);

    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j) {
            A[i*N+j] = (i+j)/(double)N;
            B[i*N+j] = i*j*3.14;
            C[i*N+j] = 0;
        }

    tbb::task_scheduler_init init(nworkers);

    ff::ffTime(ff::START_TIME);
    tbb::parallel_for(tbb::blocked_range<long>(0,N), Mult());
    printf("%d Time = %g (ms)\n", nworkers,ff::ffTime(ff::STOP_TIME));

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
