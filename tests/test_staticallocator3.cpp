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
 *  The StaticAllocator in each Source node uses the following amount of memory:
 *    #FlatMap * (qlen + 2) * sizeof(task);
 *
 *  The StaticAllocator in each FlatMap node uses the following amount of memory:
 *   (#Sink + 1) * #Map * (qlen + 2) * sizeof(task)
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

// some default values
static long qlen     = 2;     // -q 
static long howmany  = 11;    // -w
static long ntasks   = 1000;  // -n
static long nSink    = 3;     // -k
static long nFlatMap = 11;    // -f
static long nMap     = 3;     // -m
static long nSource  = 5;     // -s
static bool enablestd= false; // -a
static bool sinksleep= false; // -u
static std::mutex mtx;  // used only for pretty printing

struct Source: ff_monode_t<S_t> {
	Source(long ntasks, StaticAllocator* SAlloc):
        ntasks(ntasks), SAlloc(SAlloc) {}

    int svc_init() {
        return (!enablestd?SAlloc->init():0);
    }
	S_t* svc(S_t*) {
        long start = get_my_id()*ntasks;
        for (long i=1;i<=ntasks;++i){
            S_t* p;
            if (!enablestd) {
                // NOTE: 
                // to know which allocator to use, we must use
                // the get_next_free_channel to know which will be
                // the output channel selected by the ff_send_out
                //
                int ch = get_next_free_channel();
                assert(ch>=0 && ch <(int)get_num_outchannels());
                SAlloc->alloc(p, ch);
            } else {
                p = new S_t;
                assert(p);
            }

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

    int svc_init() {
        return (!enablestd?SAlloc->init():0);
    }

    S_t* svc(S_t* in) {
        for(int i=0;i<howmany; ++i) {
            S_t* p;
            if (!enablestd) {
                int ch = get_next_free_channel();
                assert(ch>=0 && ch <(int)get_num_outchannels());
                SAlloc->alloc(p,ch);
            } else {
                p = new S_t;
                assert(p);
            }
            
            *p = *in;
            ff_send_out(p);
        }
        if (enablestd) delete in;
        else 
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
        if (sinksleep && (get_my_id()%2) == 0) usleep(3000);  // Sinks with even id are bottleneck
        ++M[in->t];        
        // {
        //    std::lock_guard<std::mutex> lck (mtx);
        //    std::cout << "Sink" << get_my_id() << " received " << in->t << ", " << in->f << "\n";
        // }
        if (enablestd) delete in;
        else 
            StaticAllocator::dealloc(in);
        return GO_ON;
    }
    
    std::map<int,int> M;
};


static inline void usage(const char *argv0) {
    std::cout << "--------------------\n";
    std::cout << "Usage: " << argv0 << " [options]\n";
    std::cout << "\nOptions:\n";
    std::cout << " -s number of source nodes (default " << nSource << ")\n";
    std::cout << " -f number of flat map nodes (default " << nFlatMap << ")\n";
    std::cout << " -m number of map nodes (default " << nMap << ")\n";
    std::cout << " -k number of sink nodes (default " << nSink << ")\n";
    std::cout << " -n number of tasks generated by each source (default " << ntasks << ")\n";
    std::cout << " -q set the queue length (default " << qlen << ")\n";
    std::cout << " -w set how many tasks generate each flat map for one input (default " <<  howmany << ")\n";
    std::cout << " -a enable the standard allocator (new/delete) (default disabled)\n";
    std::cout << " -h print this message\n";
    std::cout << " -u enable sleeping for even sinks (default disabled)\n";
    std::cout << "--------------------\n\n";
}
static bool isNumber(const char* s, long &n) {
    try {
        size_t e;
        n=std::stol(s, &e, 10);
        return e == strlen(s);
    } catch (const std::invalid_argument&) {
        return false;
    } catch (const std::out_of_range&) {
        return false;
    }
}
int parseCommandLine(int argc, char *argv[]) {
    extern char *optarg;
    const std::string optstr="s:m:f:k:n:q:w:u:ah";
    long opt, _s=nSource, _m=nMap, _f=nFlatMap, _k=nSink, _n=ntasks, _q=qlen, _w=howmany;
    while((opt = getopt(argc, argv, optstr.c_str())) != -1) {
	switch(opt) {
	case 's': {
	    if (!isNumber(optarg, _s)) {
            std::cerr << "Error: wrong '-s' option\n";
            usage(argv[0]);
            return -1;
	    }
	} break;
	case 'm': {
	    if (!isNumber(optarg, _m)) {
            std::cerr << "Error: wrong '-m' option\n";
            usage(argv[0]);
            return -1;
	    }
	} break;
	case 'f': {
	    if (!isNumber(optarg, _f)) {
            std::cerr << "Error: wrong '-f' option\n";
            usage(argv[0]);
            return -1;
	    }        
	} break;
	case 'k': {
	    if (!isNumber(optarg, _k)) {
            std::cerr << "Error: wrong '-k' option\n";
            usage(argv[0]);
            return -1;
	    }
	} break;
	case 'n': {
	    if (!isNumber(optarg, _n)) {
            std::cerr << "Error: wrong '-n' option\n";
            usage(argv[0]);
            return -1;
	    }	    
	} break;
	case 'q': {
	    if (!isNumber(optarg, _q)) {
            std::cerr << "Error: wrong '-q' option\n";
            usage(argv[0]);
            return -1;
	    }	    	    
	} break;
	case 'w': {
	    if (!isNumber(optarg, _w)) {
            std::cerr << "Error: wrong '-w' option\n";
            usage(argv[0]);
            return -1;
	    }	    	    
	} break;
	case 'a': { enablestd=true; } break;
    case 'u': { sinksleep=true; } break;
    case 'h':
	default:
	    usage(argv[0]);
	    return -1;
	}
    }
    if (optind<argc && argv[optind][0]!='-') {
        std::cerr << "Error: option " << argv[optind] << " not recognized\n";
        usage(argv[0]);
        return -1;
    }
    nSource=_s; nMap=_m; nFlatMap=_f; nSink=_k; ntasks=_n; qlen=_q; howmany=_w;
    return 0;
}

int main(int argc, char* argv[]) {
    if (argc!=1) {
        if (parseCommandLine(argc, argv)<0) return -1;
    }
    std::cout << "running with the following settings:\n";
    std::cout << " nSource (-s)            = " << nSource << "\n"; 
    std::cout << " nFlatMap (-f)           = " << nFlatMap << "\n"; 
    std::cout << " nMap (-m)               = " << nMap << "\n"; 
    std::cout << " nSink (-k)              = " << nSink << "\n"; 
    std::cout << " ntasks (-n)             = " << ntasks << "\n"; 
    std::cout << " qlen (-q)               = " << qlen << "\n"; 
    std::cout << " howmany  (-w)           = " << howmany << "\n"; 
    std::cout << " standard allocator (-a) = " << (enablestd?"enabled":"disabled") << "\n";
    std::cout << " even sinks sleep (-u)   = " << (sinksleep?"enabled":"disabled") << "\n\n";

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
        StaticAllocator* FlatMapAlloc = nullptr;
        if (!enablestd){
            // NOTE: for each queue we have +2 slots
            FlatMapAlloc = new StaticAllocator((nSink +1)*(qlen+2), sizeof(S_t), nMap);
            assert(FlatMapAlloc);
        }
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
        StaticAllocator* SourceAlloc  = nullptr;
        if (!enablestd) {
            // NOTE: for each queue we have +2 slots
            SourceAlloc = new StaticAllocator( 1*(qlen+2), sizeof(S_t), nFlatMap);
            assert(SourceAlloc);
        }
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



