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
 *
 *   Source --->  FlatMap ---> Map ---> Sink
 *
 *
 *
 */

#include <mutex>
#include <iostream>
#include <string>
#include <ff/ff.hpp>
#include <ff/staticallocator.hpp>

using namespace ff;

struct S_t {
    int t;
    float f;
};

long qlen    = 1;
long howmany = 10;
long ntasks  = 1000;

std::mutex mtx;  // used only for pretty printing

struct Source: ff_monode_t<S_t> {
	Source(long ntasks): ntasks(ntasks) {}

    int svc_init() {
        return SAlloc->init();
    }
	S_t* svc(S_t*) {        
        long start = get_my_id()*ntasks;
        for (long i=1;i<=ntasks;++i){
            S_t* p;
            SAlloc->alloc(p);
            p->t = start+i;
            p->f = p->t*1.0;
            ff_send_out(p);
        }
		return EOS; 
	}

    long ntasks;
    StaticAllocator* SAlloc = nullptr;
};
struct FlatMap: ff_monode_t<S_t> {

    int svc_init() {
        return SAlloc->init();
    }
    S_t* svc(S_t* in) {
        for(int i=0;i<howmany; ++i) {
            S_t* p;
            SAlloc->alloc(p);
            *p = *in;
            ff_send_out(p);
        }
        StaticAllocator::dealloc(in);
        return GO_ON;
	}
    StaticAllocator *SAlloc = nullptr;
};

struct Map: ff_monode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};


struct Sink: ff_minode_t<S_t> {
    S_t* svc(S_t* in) {
        ++cnt2;
        {
            std::lock_guard<std::mutex> lck (mtx);
            std::cout << "Sink received " << in->t << ", " << in->f << "\n";
        }
        if (in->t != cnt || in->f != (cnt*1.0)) 
        {
            std::lock_guard<std::mutex> lck (mtx);
            std::cout << "Sink ERROR " << in->t << ", " << in->f << " cnt = " << cnt << "\n";
            abort();
        }
        StaticAllocator::dealloc(in);
        if (cnt2 == howmany) {
            cnt++;
            cnt2= 0;
        }
        return GO_ON;
    }

    long cnt = 1;
    long cnt2= 0;
};


int main(int argc, char* argv[]) {    
    Source  Sc(ntasks);
    FlatMap FM;
    Map     M;
    Sink    Sk;

    // NOTE: for each queue we have +2 slots
    StaticAllocator* SourceAlloc  = new StaticAllocator(qlen+2, sizeof(S_t)); 
    StaticAllocator* FlatMapAlloc = new StaticAllocator((qlen+2)*2, sizeof(S_t));

    Sc.SAlloc = SourceAlloc;
    FM.SAlloc = FlatMapAlloc;

    ff_pipeline pipeMain(false, qlen, qlen, true);

    pipeMain.add_stage(&Sc);
    pipeMain.add_stage(&FM);
    pipeMain.add_stage(&M);
    pipeMain.add_stage(&Sk);

    if (pipeMain.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }

    delete SourceAlloc;
    delete FlatMapAlloc;
    
    {
        std::lock_guard<std::mutex> lck (mtx);
        std::cout << "Done\n";
    }
    return 0;
}



