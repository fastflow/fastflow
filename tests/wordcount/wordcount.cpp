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

/* Simple test for the StaticAllocator (to use instead the standard allocator 
 * use command line option '-a').  
 *
 *
 * The Source produces a continuous stream of lines of a text file. 
 * The Splitter tokenizes each line producing a new output item for each 
 * word extracted. The Counter receives single words from the line
 * splitter and counts how many occurrences of the same word appeared
 * on the stream until that moment. The Sink node receives every result 
 * produced by the word counter and counts the total number of words.
 *            
 * One possible FastFlow graph is the following:
 *            
 *   Source-->Splitter -->| 
 *                        | --> Counter --> Sink           
 *   Source-->Splitter -->| 
 *                        | --> Counter --> Sink
 *   Source-->Splitter -->| 
 *
 *  /<---- pipe ---->/         /<-- pipe -->/
 *  /<---------------- a2a ---------------->/
 *
 *  Source and FlatMap produce more outputs for a single input, 
 *  the data items flowing into the output streams are allocated
 *  using the StaticAllocator (one for each Source and FlatMap).
 *
 *  The StaticAllocator in each Source node uses the following amount of memory:
 *   1*(qlen+2)
 *
 *  The StaticAllocator in each FlatMap node uses the following amount of memory:
 *   (1 + 1) * (qlen+2) * nSink
 *
 *
 *  Command line options:
 *   -f file : text file
 *   -p n,m  : defines the number of replicas of the pipeline Source --> Splitter (n) 
 *             and of the pipeline Counter --> Sink (m)
 *   -a      : if set, the program will use the standard allocator (optional)
 *   -t time : running time in seconds in the range 1-100 (optional, default value 15s)
 *   
 *
 */

#define FF_BOUNDED_BUFFER
#if !defined(DEFAULT_BUFFER_CAPACITY)
#define DEFAULT_BUFFER_CAPACITY 2048
#endif
#define BYKEY true

#include <iostream>
#include <iomanip> 
#include <string>
#include <sstream>
#include <vector>
#include <atomic>
#include <map>
#include <ff/ff.hpp>
#include <ff/staticallocator.hpp>

using namespace ff;
using namespace std;

const size_t qlen = DEFAULT_BUFFER_CAPACITY;
const int MAXLINE=128;
const int MAXWORD=32;
struct tuple_t {
    char text_line[MAXLINE];   // line of the parsed dataset (text, book, ...)
    size_t key;                // line number
    uint64_t id;               // id set to zero
    uint64_t ts;               // timestamp
};
struct result_t {
    char     key[MAXWORD];  // key word
    uint64_t id;            // id that indicates the current number of occurrences of the key word
    uint64_t ts;            // timestamp
};


vector<tuple_t> dataset;                    // contains all the input tuples in memory
atomic<long> total_lines=0;                   // total number of lines processed by the system
atomic<long> total_bytes=0;                   // total number of bytes processed by the system

/// application run time (source generates the stream for app_run_time seconds, then sends out EOS)
unsigned long app_run_time = 15;  // time in seconds

static inline unsigned long current_time_usecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000L + (t.tv_nsec / 1000);
}
static inline unsigned long current_time_nsecs() {
    struct timespec t;
    clock_gettime(CLOCK_REALTIME, &t);
    return (t.tv_sec)*1000000000L + t.tv_nsec;
}


struct Source: ff_monode_t<tuple_t> {
    StaticAllocator* SAlloc=nullptr;
    size_t next_tuple_idx = 0;          // index of the next tuple to be sent
    int generations       = 0;          // counts the times the file is generated
    long generated_tuples = 0;          // tuples counter
    long generated_bytes  = 0;          // bytes counter

    // time variables
    unsigned long app_start_time;   // application start time
    unsigned long current_time;

    Source(StaticAllocator* SAlloc,
           const unsigned long _app_start_time):
        SAlloc(SAlloc), app_start_time(_app_start_time),current_time(_app_start_time)  {
    }

    int svc_init() {
        return (SAlloc?SAlloc->init():0);
    }
	tuple_t* svc(tuple_t*) {
        while(1) {
            tuple_t* t;
            if (SAlloc) {
                SAlloc->alloc(t);
            } else {
                t = new tuple_t;
            }
            assert(t);

            if (generated_tuples > 0) current_time = current_time_nsecs();
            if (next_tuple_idx == 0) generations++;         // file generations counter
            generated_tuples++;                             // tuples counter
            // put a timestamp and send the tuple
            *t = dataset.at(next_tuple_idx);
            generated_bytes += sizeof(tuple_t);
            t->ts = current_time - app_start_time;
            ff_send_out(t);
            ++next_tuple_idx;
#if defined(ONESHOT)   // define it for a one shot run
            if (next_tuple_idx == dataset.size()) break;
#endif            
            next_tuple_idx %= dataset.size();
            // EOS reached
            if (current_time - app_start_time >= (app_run_time*1000000000L) && next_tuple_idx == 0) 
                break;
        }
        total_lines.fetch_add(generated_tuples);
        total_bytes.fetch_add(generated_bytes);
      	return EOS; 
	}
};
struct Splitter: ff_monode_t<tuple_t, result_t> {
    Splitter(StaticAllocator* SAlloc): SAlloc(SAlloc) {
    }

    int svc_init() {
        noutch=get_num_outchannels();
        return (SAlloc?SAlloc->init():0);
    }

    result_t* svc(tuple_t* in) {        
        char *tmpstr;
        char *token = strtok_r(in->text_line, " ", &tmpstr);
        while (token) {
#if defined(BYKEY)
            int ch = std::hash<std::string>()(std::string(token)) % noutch;
#else            
            int ch = ++idx % noutch;
#endif            
            result_t* r;
            if (SAlloc) {
                SAlloc->alloc(r,ch);
            } else {
                r = new result_t;
            }
            assert(r);
            strncpy(r->key, token, MAXWORD-1);
            r->key[MAXWORD-1]='\0';
            r->ts  = in->ts;

            ff_send_out_to(r, ch);
            token = strtok_r(NULL, " ", &tmpstr);
        }

        if (SAlloc)        
            StaticAllocator::dealloc(in);
        else
            delete in;
        
        return GO_ON;
	}
    StaticAllocator* SAlloc=nullptr;
    long noutch=0;
    long idx=-1;
};

struct Counter: ff_minode_t<result_t> {
    result_t* svc(result_t* in) {
        ++M[std::string(in->key)];
        // number of occurrences of the string word up to now
        in->id = M[std::string(in->key)]; 
        return in;
    }
    size_t unique() {
        // std::cout << "Counter:\n";
        //  for (const auto& kv : M)  {
        //      std::cout << kv.first << " --> " << kv.second << "\n";
        //  }
        return M.size();
    }
    std::map<std::string,size_t> M;
};

struct Sink: ff_node_t<result_t> {
    Sink(bool usestd):usestd(usestd) {}
    
    result_t* svc(result_t* in) {
        ++words;
        if (usestd) delete in;
        else 
            StaticAllocator::dealloc(in);
        return GO_ON;
    }
    size_t words=0;
    bool usestd;
};


/** 
 *  @brief Parse the input file and create all the tuples
 *  
 *  The created tuples are maintained in memory. The source node will generate the stream by
 *  reading all the tuples from main memory.
 *  
 *  @param file_path the path of the input dataset file
 */ 
int parse_dataset_and_create_tuples(const string& file_path) {
    ifstream file(file_path);
    if (file.is_open()) {
        size_t all_records = 0;         // counter of all records (dataset line) read
        string line;
        while (getline(file, line)) {
            // process file line
            if (!line.empty()) {
                if (line.length() > MAXLINE) {
                    std::cerr << "ERROR INCREASE MAXLINE\n";
                    return -1;
                }
                tuple_t t;
                strncpy(t.text_line, line.c_str(), MAXLINE-1);
                t.text_line[MAXLINE-1]='\0';
                t.key = all_records;
                t.id  = 0;
                t.ts  = 0;
                all_records++;
                dataset.push_back(t);
            }
        }
        file.close();
    } else {
        std::cerr << "Unable to open the file " << file_path << "\n";
        return -1;
    }
    return 0;
}


int main(int argc, char* argv[]) {
    /// parse arguments from command line
    std::string file_path("");
    size_t source_par_deg = 0;
    size_t sink_par_deg = 0;
    bool usestd = false;
    
    if (argc >= 3) {
        int option = 0;
        while ((option = getopt(argc, argv, "f:p:t:a")) != -1) {
            switch (option) {
            case 'f': file_path=string(optarg);  break;
            case 'p': {
                vector<size_t> par_degs;
                string pars(optarg);
                stringstream ss(pars);
                for (size_t i; ss >> i;) {
                    par_degs.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                if (par_degs.size() != 2) {
                    std::cerr << "Error in parsing the input arguments -p, the format is n,m\n";
                    return -1;
                }
                else {
                    source_par_deg = par_degs[0];
                    sink_par_deg = par_degs[1];
                }
                break;
            }
            case 'a': { usestd=true; } break;
            case 't': {
                long t = stol(optarg);
                if (t<=0 || t > 100) {
                    std::cerr << "Wrong value for the '-t' option, it should be in the range [1,100]\n";
                    return -1;
                }
                app_run_time = t;
            } break;
            default: {
                std::cerr << "Error in parsing the input arguments\n";
                return -1;
            }
            }
        }
    } else {
        std::cerr << "Parameters:  -p  <nSource/nSplitter,nCounter/nSink> -f <filepath> [-a -t n]\n";
        return -1;
    }
    if (file_path.length()==0) {
        std::cerr << "The file path is empty, please use option -f <filepath>\n";
        return -1;
    }
    if (source_par_deg == 0 || sink_par_deg==0) {
        std::cerr << "Wrong values for the parallelism degree, please use option -p <nSource/nSplitter, nCounter/nSink>\n";
        return -1;
    }

    cout << "Executing WordCount with parameters:" << endl;
    cout << "  * queues length  : " << DEFAULT_BUFFER_CAPACITY << endl;
    cout << "  * source/splitter: " << source_par_deg << endl;
    cout << "  * counter/sink   : " << sink_par_deg << endl;
    cout << "  * running time   : " << app_run_time << " (s)\n";
    if (usestd) cout << " USING STANDARD ALLOCATOR\n";
    cout << "  * topology: source -> splitter -> counter -> sink" << endl;

    /// data pre-processing
    parse_dataset_and_create_tuples(file_path);
    /// application starting time
    unsigned long app_start_time = current_time_nsecs();

    std::vector<Counter*> C(sink_par_deg);    
    std::vector<Sink*> S(sink_par_deg);    
    std::vector<ff_node*> L;
    std::vector<ff_node*> R;
    
    for (size_t i=0;i<source_par_deg; ++i) {        
        StaticAllocator* SourceAlloc  = nullptr;
        StaticAllocator* SplitterAlloc = nullptr;
        
        if (!usestd) {
            // NOTE: for each queue we have +2 slots
            SourceAlloc = new StaticAllocator( 1*(qlen+2), std::max(sizeof(tuple_t),sizeof(result_t)), 1);
            assert(SourceAlloc);
            SplitterAlloc = new StaticAllocator((1 + 1)*(qlen+2), std::max(sizeof(tuple_t),sizeof(result_t)), sink_par_deg);
            assert(SplitterAlloc);
            
        }
        ff_pipeline* pipe0 = new ff_pipeline(false, qlen, qlen, true);
        
        pipe0->add_stage(new Source((usestd?nullptr:SourceAlloc), app_start_time), true);
        pipe0->add_stage(new Splitter((usestd?nullptr:SplitterAlloc)), true);
        L.push_back(pipe0);
    }
    for (size_t i=0;i<sink_par_deg; ++i) {
        ff_pipeline* pipe1 = new ff_pipeline(false, qlen, qlen, true);
        S[i] = new Sink(usestd);
        C[i] = new Counter;
        pipe1->add_stage(C[i]);
        pipe1->add_stage(S[i]);
        R.push_back(pipe1);
    }

    ff_a2a a2a(false, qlen, qlen, true);
    a2a.add_firstset(L, 0, true);
    a2a.add_secondset(R, true);
    ff_pipeline pipeMain(false, qlen, qlen, true);
    pipeMain.add_stage(&a2a);

    std::cout << "Starting " << pipeMain.numThreads() << " threads\n";
    cout << "Executing topology" << endl;
    /// evaluate topology execution time
    volatile unsigned long start_time_main_usecs = current_time_usecs();
    if (pipeMain.run_and_wait_end()<0) {
        error("running pipeMain\n");
        return -1;
    }
    volatile unsigned long end_time_main_usecs = current_time_usecs();
    cout << "Exiting" << endl;
    double elapsed_time_seconds = (end_time_main_usecs - start_time_main_usecs) / (1000000.0);
    cout << "elapsed time     : " << elapsed_time_seconds << "(s)\n";
    double throughput = total_lines / elapsed_time_seconds;
    double mbs = (double)((total_bytes / 1048576) / elapsed_time_seconds);
    cout << "Measured throughput: " << (int) throughput << " lines/second, " << mbs << " MB/s" << endl;
    cout << "total_lines sent : " << total_lines << "\n";
    cout << "total_bytes sent : " << std::setprecision(3) << total_bytes/1048576.0 << "(MB)\n";

    
#if 1
    size_t words=0;
    size_t unique=0;
    for(size_t i=0;i<S.size();++i) {
        words += S[i]->words;
        unique+= C[i]->unique();
        delete S[i];
        delete C[i];
    }
    cout << "words            : " << words << "\n";
    cout << "unique           : " << unique<< "\n";
#endif
    return 0;
}
