/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/* square matrices */

#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>

#include <ff/map.hpp>

using namespace ff;

#define NUM_ITER   10

double* IN  = NULL;
double* OUT = NULL;


void printMatrix(double* M, const size_t size) {
    for(size_t i=0;i<size;++i) {
	for(size_t j=0;j<size;++j)
	    printf("%.4f ", M[i*size+j]);
	printf("\n");
    }
    printf("\n");
}

struct ab_t {
    ab_t():a(0.0),b(0.0) {}
    ab_t(double a, double b):a(a),b(b) {}
    double a;
    double b;
};

class Emitter:public ff_node {
protected:
    void generate_stream() {
        for(size_t i=1;i<(size+1);++i) 
            ff_send_out((void*)i);
    }
public:
    Emitter(const size_t size):size(size),k(0) {}
    void * svc(void* task) {
        if (task==NULL) {
            generate_stream();
            k=1;
            return GO_ON;
        }
        ab_t* ab = (ab_t*)task;
        //printMatrix(IN,size+2);
        //printMatrix(OUT,size+2);
        printf("a=%.4f  b=%.4f  (b-a)=%.4f\n", ab->a, ab->b, ab->b-ab->a);
        std::swap(IN,OUT);
        if (++k > NUM_ITER) { delete ab; return NULL;}
        generate_stream();
        delete ab;
        return GO_ON;
    }
protected:
    const size_t size;
    long k;
};

class Collector:public ff_node {
public:
    Collector(const size_t size):size(size),cnt(0) {}
    void * svc(void* task) {
        ab_t* t =(ab_t*)(task);
        ab.a += t->a, ab.b += t->b;
        if (++cnt >= size) {
            ff_send_out(new ab_t(ab));
            cnt=0;
            ab.a=0, ab.b=0;
        }
        delete t;
        return GO_ON;
    }
protected:
    const size_t size;
    size_t cnt;
    ab_t   ab;
};

class Worker:public ff_node {
public:
    Worker(const size_t size):size(size),rowsize(size+2) {}

    void * svc(void * task) {
        size_t i = (size_t)task;
        double sumIN=0.0, sumOUT=0.0;

        for(size_t j=1;j<(size+1);++j) {
            OUT[i*rowsize+j] = sin(IN[i*rowsize+j]) + 
                sin(IN[i*rowsize+(j-1)]) + sin(IN[i*rowsize+(j+1)]); 
#if defined(REDUCE)
            sumIN += IN[i*rowsize+j];
#endif
        }
        for(size_t j=1;j<(size+1);++j) {
            OUT[i*rowsize+j] += sin(IN[(i-1)*rowsize+j]) + sin(IN[(i+1)*rowsize+j]);
            OUT[i*rowsize+j] /= 5.0;
#if defined(REDUCE)
            sumOUT += OUT[i*rowsize+j];
#endif
        }
        return (new ab_t(sumIN,sumOUT));
    }
protected:
    const size_t size;
    const size_t rowsize;
};


int main(int argc, char* argv[]) {
    if (argc<3) {
        printf("use: %s matsize nworkers\n", argv[0]);
        return -1;
    }

    size_t size=atoi(argv[1]);
    int    nw  =atoi(argv[2]);

    IN  = new double[(size+2)*(size+2)];
    OUT = new double[(size+2)*(size+2)];

    /* init */
    const size_t rowsize=size+2;
    bzero(IN, rowsize*rowsize*sizeof(double));
    bzero(OUT,rowsize*rowsize*sizeof(double));
    for(size_t i=1;i<(size+1);++i)
	for(size_t j=1;j<(size+1);++j) {
	    IN[i*rowsize+j] = i+((double)(j)/(rowsize*3.14));
	    OUT[i*rowsize+j] = 0.0;
	}

#if defined(USE_OPENMP)
    long k=0;
    double a,b;
    do {
#pragma omp parallel for
        for(size_t i=1;i<(size+1);++i)
            for(size_t j=1;j<(size+1);++j)
                OUT[i*rowsize+j] = sin(IN[i*rowsize+j]) + 
                    sin(IN[i*rowsize+(j-1)]) + sin(IN[i*rowsize+(j+1)]);
        
#pragma omp parallel for
        for(size_t i=1;i<(size+1);++i)
            for(size_t j=1;j<(size+1);++j) {
                OUT[i*rowsize+j] += sin(IN[(i-1)*rowsize+j]) + sin(IN[(i+1)*rowsize+j]);
                OUT[i*rowsize+j] /= 5.0;
            }

        a=0.0,b=0.0;
#if defined(REDUCE)

#pragma omp parallel for reduction(+ : a)
        for(size_t i=0;i<rowsize*rowsize;++i)
            a += IN[i];
#pragma omp parallel for reduction(+ : b)
        for(size_t i=0;i<rowsize*rowsize;++i)
            b += OUT[i];
        //printf("a=%.4f  b=%.4f  (b-a)=%.4f\n", a, b, b-a);
#endif // REDUCE 

        //printMatrix(IN,size+2);
        //printMatrix(OUT,rowsize);
        
        std::swap(IN,OUT);
    } while(++k<NUM_ITER);

                
#else 
    ff_farm<> farm;
    std::vector<ff_node*> w;
    for(int i=0;i<nw;++i)
        w.push_back(new Worker(size));
    farm.add_workers(w);
    farm.add_emitter(new Emitter(size));
    farm.add_collector(new Collector(size));
    farm.wrap_around();
    farm.run_and_wait_end();

#endif
    
    

    if (IN)  delete [] IN;
    if (OUT) delete [] OUT;

    return 0;

}
