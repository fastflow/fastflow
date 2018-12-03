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

/*
 * Very basic test for the FastFlow pipeline (3 stages).
 *
 */

#include <cmath>
#include <iostream>
#include <ff/ff.hpp>
using namespace ff;

int stage1=0;
int stage2=1;
int stage3=2;


inline double f1(unsigned long a) { return a * 3.1415; }
inline double f2(double t)        { return (1/((2.0)*t) + atan(1.0/t))/sin(1/t); }
inline double f3(double t)        { return (t*sin(t)/tan(t)) + (sin(t)*cos(1/t)/tan(1/t) + tan(1/t)*atan(t)*cos(t))*3.1415; }
// generic stage
class Stage1: public ff_node {
public:
    Stage1(unsigned long streamlen):streamlen(streamlen)  {}

    int svc_init() {
        if (ff_mapThreadToCpu(stage1)!=0)
            printf("Cannot map Stage1 to CPU %d\n",stage1);
        t = (double*)malloc(sizeof(double)*streamlen);
        return 0;
    }

    void * svc(void *) {
        for(long i=streamlen;i>0;--i) {
            t[i-1]=f1(i);
            ff_send_out(&t[i-1]);
        }
        return NULL;
    }
private:
    double * t;
    unsigned long streamlen;
};

class Stage2: public ff_node {
public:
    int svc_init() {
        if (ff_mapThreadToCpu(stage2)!=0)
            printf("Cannot map Stage2 to CPU %d\n",stage1);
        return 0;
    }

    void * svc(void * task) {
        double * t = (double *)task;        
        if (t) *t = f2(*t)+f2(1/(*t)); 
        return t;
    }
};

class Stage3: public ff_node {
public:
    int svc_init() {
        if (ff_mapThreadToCpu(stage3)!=0)
            printf("Cannot map Stage3 to CPU %d\n",stage1);
        somma=0.0;
        return 0;
    }
    
    void * svc(void * task) {
        double * t = (double *)task;
        
        if (t) {
            *t=f3(*t)*f1(2.0*(*t)); 
            somma += *t;
        }
        return GO_ON;
    }
    void svc_end() {
        printf("%g\n",somma);
    }
private:
    double somma;
};

int main(int argc, char * argv[]) {
    int seq = 0;
    int streamlen = 10000;
    if (argc>1) {
        if (argc<6 && (argc==3 && atoi(argv[1]))<=0) {
            std::cerr << "use: "  << argv[0] << " 0|1 streamlen c1 c2 c3\n";
            std::cerr << "    0 pipeline version\n";
            std::cerr << "    1 sequential version\n";
            std::cerr << " c1-3 are core ids for the 3 pipeline stages\n";
            return -1;
        }
        
        seq = atoi(argv[1]);
        streamlen=atoi(argv[2]);
        if (argc>3) {
            stage1 = atoi(argv[3]);
            stage2 = atoi(argv[4]);
            stage3 = atoi(argv[5]);
        }
    }
    if (seq) {
        ffTime(START_TIME);

        double* t = (double*)malloc(sizeof(double)*streamlen);
        double somma=0.0;
        for(long i=streamlen;i>0;--i) {
            t[i-1] =f1(i);
            t[i-1] = f2(t[i-1])+f2(1/(t[i-1])); 
            t[i-1]=f3(t[i-1])*f1(2*(t[i-1])); 
            somma += t[i-1];
        }
        ffTime(STOP_TIME);
        printf("%f\n",somma);
        std::cerr << "DONE, seq time= " << ffTime(GET_TIME) << " (ms)\n";   
        return 0;
    }
    
    // build a 2-stage pipeline
    ff_pipeline pipe(false,1024,1024,false);
    pipe.add_stage(new Stage1(streamlen));
    pipe.add_stage(new Stage2());
    pipe.add_stage(new Stage3());

    ffTime(START_TIME);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }
    ffTime(STOP_TIME);

    std::cerr << "DONE, work  time= " << pipe.ffwTime() << " (ms)\n";
    std::cerr << "DONE, pipe  time= " << pipe.ffTime() << " (ms)\n";
    std::cerr << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
    pipe.ffStats(std::cerr);
    return 0;
}
