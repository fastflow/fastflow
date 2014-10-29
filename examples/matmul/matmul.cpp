#include <cstdio>
#include <cstdlib>
#include <ff/utils.hpp>

static long    N=0;      // matrix size
static double* A=NULL;
static double* B=NULL;
static double* C=NULL;

void init_dense() {
    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j) {
            A[i*N+j] = (i+j)/(double)N;
            B[i*N+j] = i*j*3.14;
            C[i*N+j] = 0;
        }
}

void init_sparse(long k) {

    // init B and C
    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j) {
            B[i*N+j] = i*j*3.14;
            C[i*N+j] = 0;
        }

    // init A
    int K = k;
    for(long i=0;i<k;++i)  {
        for(long j=0;j<K;++j)
            A[i*N+j] = (i+j)/(double)N;
	--K;
    }
    K = k;
    for(long i=(N-1);i>=(N-k);--i)  {
        for(long j=(N-1);j>=(N-K);--j)
            A[i*N+j] = (i+j)/(double)N;
	--K;
    }
}


int main(int argc, char* argv[]) {
    if (argc<2) {
        printf("use: %s size [sparse-factor 0<k<size]\n", argv[0]);
        return -1;
    }
    N               =atol(argv[1]);
    assert(N>0);
    long k=-1;
    if (argc==3) k=atol(argv[2]);

    A = (double*)malloc(N*N*sizeof(double));
    B = (double*)malloc(N*N*sizeof(double));
    C = (double*)malloc(N*N*sizeof(double));
    assert(A && B && C);

    if (k>0 && k<N)
	init_sparse(k);
    else 
	init_dense();
#if 0
    for(long i=0;i<N;++i) {
        for(long j=0;j<N;++j) 
	    printf("%.2g ", A[i*N+j]);
	printf("\n");
    }
#endif	    

  ff::ffTime(ff::START_TIME);
#if defined(OPTIMIZE_CACHE)
#if defined(USE_OPENMP)
    #pragma omp parallel for schedule(auto)
#endif
    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j)
            for(long k=0;k<N;++k)
                C[j*N+k] += A[j*N+i]*B[i*N+k];
#else // OPTIMIZE_CACHE
#if defined(USE_OPENMP)
    #pragma omp parallel for schedule(auto)
#endif
    for(long i=0;i<N;++i) 
        for(long j=0;j<N;++j)
            for(long k=0;k<N;++k)
                C[i*N+j] += A[i*N+k]*B[k*N+j];
#endif // OPTIMIZE_CACHE
    printf("Time = %g (ms)\n", ff::ffTime(ff::STOP_TIME));
    return 0;
}


