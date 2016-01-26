#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include <ff/stencilReduceOCL.hpp>
using namespace ff;

// in place
FF_OCL_STENCIL_ELEMFUNC_ENV(mapfenv, float, size, k, M, env_t, env,
			 (void)size;
			 return env->coeff * M[k];
);



struct env_t {
    env_t() : coeff(0), n_elems(0) {}
    env_t(long n_elems, long coeff):n_elems(n_elems),coeff(coeff) {}
    long   n_elems;
    long   coeff;
};

struct Task {
    Task(float *M, long coeff, long n_elems):
        M(M),env(n_elems,coeff) {};
    float       *M;
    env_t        env; // contains 'coeff' and 'n_elems'
};

struct oclTaskEnv: public baseOCLTask<Task, float, float> {
	oclTaskEnv() {}
    void setTask(Task *t) {
        assert(t);
	setInPtr(t->M, t->env.n_elems);
        setOutPtr(t->M);                  // in place !!
        setEnvPtr(&t->env, 1);
    }
};

int main(int argc, char * argv[]) {
    size_t size=1024;
    if(argc>1) size     =atol(argv[1]);
    printf("arraysize = %ld\n", size);

    float *M        = new float[size];
    for(size_t j=0;j<size;++j) M[j]=j;

    Task task(M, 2, (long)size);
    ff_mapOCL_1D<Task, oclTaskEnv> mapOCLenv(task, mapfenv);
    mapOCLenv.run_and_wait_end();


#if defined(CHECK)
    for(size_t i=0;i<size;++i)
        printf("%g ", M[i]);
    printf("\n");
#endif

    delete [] M;
    return 0;
}
