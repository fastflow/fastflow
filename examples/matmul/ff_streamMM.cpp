#include <cstdlib>
#include <cstdio>
#include <vector>
#include <ff/farm.hpp>

using namespace ff;

/* globals */
double* A = NULL;
double* B = NULL;
double* C = NULL;

void Print(int N, int size) {
    int tsize=size*size;
    for(int i=0;i<N;++i) {
	for(int j=0;j<size;++j) {
	    for(int k=0;k<size;++k) {
		printf("%f ", 		C[i*tsize + j*size + k]);
	    }
	    printf("\n");
	}
	printf("\n\n");
    }
}

class Emitter: public ff_node {
public:
    Emitter(long ntasks):ntasks(ntasks) {}
    void* svc(void*) {
	for(long i=1;i<=ntasks;++i)
	    ff_send_out((void*)i);
	return NULL;
    }
private:
    long ntasks;
};

class Worker: public ff_node {
public:
    Worker(long size):size(size),tsize(size*size) {
    }
    
    void* svc(void* task) {
	long taskid = (long)(long*)task;
	--taskid;

	double* _A = &A[taskid*tsize];
	double* _B = &B[taskid*tsize];
	double* _C = &C[taskid*tsize];

	for (register int i = 0; i < size; ++i)
	    for (register int j = 0; j < size; ++j) {
		double _Ctmp=0.0;
		for (register int k = 0; k < size; ++k)
		    _Ctmp += _A[(i * size) + k] * _B[(k * size) + j];
		_C[(i * size) + j] = _Ctmp;
	    }
	return GO_ON;
    }
public:
    long size;
    long tsize;
};

int main(int argc, char* argv[]) {

    if (argc<4) {
	printf("use: %s matrix-size matrices-per-worker nworkers\n", argv[0]);
	return -1;
    }

    int size     = atoi(argv[1]);
    int mxw      = atoi(argv[2]);
    int nworkers = atoi(argv[3]);

    int N     = mxw * nworkers;
    int tsize = size*size;

    A = (double*)calloc(N, tsize*sizeof(double));
    B = (double*)calloc(N, tsize*sizeof(double));
    C = (double*)calloc(N, tsize*sizeof(double));

    for(int i=0;i<N;++i)
	for(int j=0;j<size;++j)
	    for(int k=0;k<size;++k) {
		A[i*tsize + j*size + k] = (i+j+k)/3.14;
		B[i*tsize + j*size + k] = (i+j*k) + 3.14;
	    }

    ff_farm<> farm;
    Emitter E(N);
    farm.add_emitter(&E);
    std::vector<ff_node *> w;
    for(int i=0;i<nworkers;++i) w.push_back(new Worker(size));
    farm.add_workers(w);    

    farm.run_and_wait_end();
    printf("Total time %0.2f (ms)\n", farm.ffTime());
    for(int i=0;i<nworkers;++i)
	printf("Worker %d mean task time %.2f\n", i,
	       diffmsec( ((Worker*)w[i])->getwstoptime(), ((Worker*)w[i])->getwstartime())/mxw);
    //printf("-------------------\n");Print(N,size);
    return 0;
}
