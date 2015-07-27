#include <ff/mapOCL.hpp>
using namespace ff;

// in place
FF_ARRAY_CENV(mapfenv, float, M, size, k, env_t, env,
	      (void)size;
	      return env->coeff * M[k];
);

FFREDUCEFUNC(reducef, float, x, y,
             return (x+y);
             );

struct env_t {
    env_t() {}
    env_t(long n_elems, long coeff):n_elems(n_elems),coeff(coeff) {}
    long   n_elems;
    long   coeff;
};

struct Task {
    Task(float *M, long coeff, long n_elems):
        M(M),env(n_elems,coeff), result(0.) {};
    float       *M;
    env_t        env; // contains 'coeff' and 'n_elems'
    float         result;
};

struct oclTaskEnv: public baseOCLTask<Task, float, float> {
	oclTaskEnv() {}
    void setTask(const Task *t) {
        assert(t);
        setInPtr(t->M, t->env.n_elems);
        setOutPtr(t->M);
        setEnvPtr(&t->env, 1);
        setReduceVar(&(t->result));
    }
};

int main(int argc, char * argv[]) {
    size_t size=1024;
    if(argc>1) size     =atol(argv[1]);
    printf("arraysize = %ld\n", size);

    float *M        = new float[size];
    for(size_t j=0;j<size;++j) M[j]=j;

    Task task(M, 2, (long)size);
    ff_mapreduceOCL<Task, oclTaskEnv> oclMR(task, mapfenv, reducef);
    oclMR.run_and_wait_end();

#if defined(CHECK)
    printf("%g \n", task.result);
#endif

    delete [] M;
    return 0;
}
