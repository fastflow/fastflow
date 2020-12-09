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
 *  In this test both Source and FlatMap nodes use ff_send_out;
 *
 *            |---> FlatMap-->|
 *   Source-->|               |---> Map-->|---> Sink 
 *            |---> FlatMap-->|           |
 *   Source-->|               |---> Map-->|
 *            |---> FlatMap-->|           |
 *   Source-->|               |---> Map-->|---> Sink           
 *            |---> FlatMap-->|
 *                               /<----- a2a ----->/
 *            /<--------------- a2a -------------->/
 *   /<------------------- a2a ------------------->/
 *
 *  Source and FlatMap produce more outputs for a single input, 
 *  the data items flowing into the output streams are allocated
 *  using the StaticAllocator (one for each Source and FlatMap).
 *
 */

#include <map>
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

static long qlen    = 2;
static long howmany = 11;
static long ntasks  = 100;
static std::mutex mtx;  // used only for pretty printing

struct Source: ff_monode_t<S_t> {
	Source(long ntasks, StaticAllocator* SAlloc):
        ntasks(ntasks), SAlloc(SAlloc) {}
    
	S_t* svc(S_t*) {
        long start = get_my_id()*ntasks;
        for (long i=1;i<=ntasks;++i){
            S_t* p;

            // NOTE: 
            // to know which allocator to use, we must use
            // the get_next_free_channel to know which will be
            // the output channel selected by the ff_send_out
            //
            int ch = get_next_free_channel();

            assert(ch>=0 && ch <(int)get_num_outchannels());
            SAlloc->alloc(p, ch);

            p->t = start+i;
            p->f = p->t*1.0;

            ff_send_out(p);
        }
		return EOS; 
	}

    long ntasks;
    StaticAllocator* SAlloc=nullptr;
};
struct FlatMap: ff_monode_t<S_t> {
    FlatMap(StaticAllocator* SAlloc): SAlloc(SAlloc) {
    }
    S_t* svc(S_t* in) {
        for(int i=0;i<howmany; ++i) {
            S_t* p;
            int ch = get_next_free_channel();
            assert(ch>=0 && ch <(int)get_num_outchannels());
            SAlloc->alloc(p,ch);

            *p = *in;

            ff_send_out(p);
        }
        StaticAllocator::dealloc(in);        
        return GO_ON;
	}
    StaticAllocator* SAlloc=nullptr;
};

struct Map: ff_monode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};


struct miHelper: ff_minode_t<S_t> {
    S_t* svc(S_t* in) { return in; }
};

struct Sink: ff_minode_t<S_t> {
    S_t* svc(S_t* in) {
        if ((get_my_id()%2) == 0) usleep(2000);  // even Sinks are bottleneck
        ++M[in->t];
        {
             std::lock_guard<std::mutex> lck (mtx);
             std::cout << "Sink" << get_my_id() << " received " << in->t << ", " << in->f << "\n";
        }
        StaticAllocator::dealloc(in);
        return GO_ON;
    }
    
    std::map<int,int> M;
    StaticAllocator* SAlloc=nullptr;
};


int main() {
    const int nSink    = 3;
    const int nFlatMap = 11;
    const int nMap     = 3;
    const int nSource  = 5;

    std::vector<Sink*> S(nSink);
    
    std::vector<ff_node*> L;
    std::vector<ff_node*> R;
    
    ff_a2a _1(false, qlen, qlen, true);
    for(int i=0;i<nMap;++i) 
        L.push_back(new ff_comb(new miHelper, new Map, true, true));
    for (int i=0;i<nSink;++i) {
        S[i] = new Sink;
        R.push_back(S[i]);
    }

    _1.add_firstset(L);
    _1.add_secondset(R);
    
    L.clear();
    R.clear();
    
    ff_a2a _2(false, qlen, qlen, true);
    for (int i=0;i<nFlatMap;++i) {
        StaticAllocator* FlatMapAlloc = new StaticAllocator(((1+nSink)*qlen + (1+1+nSink))*nMap, sizeof(S_t), nMap);        
        L.push_back(new ff_comb(new miHelper, new FlatMap(FlatMapAlloc), true, true));
    }
    ff_pipeline pipe1(false, qlen,qlen,true);
    pipe1.add_stage(&_1);
    R.push_back(&pipe1);

    _2.add_firstset(L);
    _2.add_secondset(R);

    L.clear();
    R.clear();
    
    ff_a2a _3(false, qlen, qlen, true);
    for (int i=0;i<nSource;++i) {
        StaticAllocator* SourceAlloc  = new StaticAllocator( (qlen +2 )*nFlatMap, sizeof(S_t), nFlatMap); 
        L.push_back(new Source(ntasks, SourceAlloc));
    }
    
    ff_pipeline pipe2(false, qlen,qlen,true);
    pipe2.add_stage(&_2);
    R.push_back(&pipe2);
    
    _3.add_firstset(L);
    _3.add_secondset(R);

    L.clear();
    R.clear();
    
    ff_pipeline pipeMain(false, qlen, qlen, true);

    pipeMain.add_stage(&_3);

    std::cout << "Starting " << pipeMain.numThreads() << " threads\n";

    unsigned long before=getusec();
    if (pipeMain.run_and_wait_end()<0) {
        error("running pipe\n");
        return -1;
    }
    std::cout << "Time= " << (getusec()-before)/1000.0 << " (ms)\n";
    // checking result
    for(int i=1; i<=ntasks*nSource;++i){
        int sum = 0;
        for(int j=0; j<nSink; ++j)
            sum += S[j]->M[i];
        if (sum != howmany) {
            std::cerr << "ERROR: wrong result!\n";
            return -1;
        }
    }
    std::cout << "RESULT OK!\n";
    return 0;
}



