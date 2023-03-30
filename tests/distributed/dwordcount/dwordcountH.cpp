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
 *  /<---- combine ---->/   /<--- combine --->/
 *  /<----------------- a2a ------------------>/
 *
 *  G1: pipe1
 *  G2: pipe2
 *
 */

#define FF_BOUNDED_BUFFER
//#define MAKE_VALGRIND_HAPPY
//#define MANUAL_SERIALIZATION
#define DEFAULT_BUFFER_CAPACITY 2048
#define BYKEY true

#include <iostream>
#include <iomanip> 
#include <string>
#include <sstream>
#include <vector>
#include <atomic>
#include <map>
#include <ff/dff.hpp>

using namespace ff;

const size_t qlen = DEFAULT_BUFFER_CAPACITY;
const int MAXLINE=128;    // character per line (CPL), a typically value is 80 CPL
const int MAXWORD=32;

struct tuple_t {
    char     text_line[MAXLINE];  // parsed line
    size_t   key;                 // line number
    uint64_t id;                  // id set to zero
    uint64_t ts;                  // timestamp
};

struct result_t {
    char     key[MAXWORD];    // key word
    uint64_t id;              // indicates the current number of occurrences of the word
    uint64_t ts;              // timestamp

	template<class Archive>
	void serialize(Archive & archive) {
		archive(key,id,ts);
	}

};

#if defined(MANUAL_SERIALIZATION)
template<typename Buffer>
void serialize(Buffer&b, result_t* input){
    b = {reinterpret_cast<char*>(input), sizeof(result_t)};
}

template<typename Buffer>
void deserialize(const Buffer&b, result_t*& strPtr){
    strPtr = reinterpret_cast<result_t*>(b.first);
}
#endif

std::vector<tuple_t> dataset;     // contains all the input tuples in memory
std::atomic<long> total_lines=0;  // total number of lines processed by the system
std::atomic<long> total_bytes=0;  // total number of bytes processed by the system


std::atomic<size_t> global_words = 0;
std::atomic<size_t> global_unique = 0;


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

struct Source: ff_node_t<tuple_t> {
    size_t next_tuple_idx = 0;          // index of the next tuple to be sent
    int generations       = 0;          // counts the times the file is generated
    long generated_tuples = 0;          // tuples counter
    long generated_bytes  = 0;          // bytes counter

    // time variables
    unsigned long app_start_time;   // application start time
    unsigned long current_time;

    Source(const unsigned long _app_start_time):
        app_start_time(_app_start_time),current_time(_app_start_time)  {
    }

	tuple_t* svc(tuple_t*) {
        while(1) {
            tuple_t* t = new tuple_t;
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

            next_tuple_idx %= dataset.size();
            // EOS reached
            if (current_time - app_start_time >= (app_run_time*1000000000L) && next_tuple_idx == 0)
                break;
        }

        std::cout << "Time expired\n";
        
        total_lines.fetch_add(generated_tuples);
        total_bytes.fetch_add(generated_bytes);
        return EOS;
	}
};
struct Splitter: ff_monode_t<tuple_t, result_t> {
    int svc_init() {
        noutch=get_num_outchannels(); // number of output channels
        return 0;
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
            result_t* r = new result_t;
            assert(r);
#if defined(MAKE_VALGRIND_HAPPY)
            bzero(r->key, MAXWORD);
#endif            
            strncpy(r->key, token, MAXWORD-1);
            r->key[MAXWORD-1]='\0';
            r->ts  = in->ts;
            r->id  = in->id;
            
            ff_send_out_to(r, ch);
            token = strtok_r(NULL, " ", &tmpstr);
        }
        delete in;        
        return GO_ON;
	}
    long noutch=0;
    long idx=-1;
};

struct Counter: ff_minode_t<result_t> {
    result_t* svc(result_t* in) {
        ++M[std::string(in->key)];
        // number of occurrences of the word up to now
        in->id = M[std::string(in->key)]; 
        return in;
    }
    void svc_end(){
        global_unique += M.size();
    }
    std::map<std::string,size_t> M;
};

struct Sink: ff_node_t<result_t> {
    result_t* svc(result_t* in) {    
        ++words;
        delete in;
        return GO_ON;
    }

    void svc_end(){
        global_words += words;
    }

    size_t words=0; // total number of words received
};

/** 
 *  @brief Parse the input file and create all the tuples
 *  
 *  The created tuples are maintained in memory. The source node will generate the stream by
 *  reading all the tuples from main memory.
 *  
 *  @param file_path the path of the input dataset file
 */ 
int parse_dataset_and_create_tuples(const std::string& file_path) {
    std::ifstream file(file_path);
    if (file.is_open()) {
        size_t all_records = 0;         // counter of all records (dataset line) read
        std::string line;
        while (getline(file, line)) {
            // process file line
            if (!line.empty()) {
                if (line.length() > MAXLINE) {
                    std::cerr << "ERROR INCREASE MAXLINE\n";
                    exit(EXIT_FAILURE);
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
    if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}
    
    /// parse arguments from command line
    std::string file_path("");
    size_t source_par_deg = 0;
    size_t sink_par_deg = 0;
    
    if (argc >= 3 || argc == 1) {
        int option = 0;    
        while ((option = getopt(argc, argv, "f:p:t:")) != -1) {
            switch (option) {
            case 'f': file_path=std::string(optarg);  break;
            case 'p': {
                std::vector<size_t> par_degs;
                std::string pars(optarg);
                std::stringstream ss(pars);
                for (size_t i; ss >> i;) {
                    par_degs.push_back(i);
                    if (ss.peek() == ',')
                        ss.ignore();
                }
                if (par_degs.size() != 2) {
                    std::cerr << "Error in parsing the input arguments -p, the format is n,m\n";
                    return -1;
                } else {
                    source_par_deg = par_degs[0];
                    sink_par_deg = par_degs[1];
                }                
            } break;
            case 't': {
                long t = std::stol(optarg);
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
        std::cerr << "Parameters:  -p  <nSource/nSplitter,nCounter/nSink> -f <filepath>\n";
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

    //if (DFF_getMyGroup() == "G1") {
        /// data pre-processing
        if (parse_dataset_and_create_tuples(file_path)< 0)
            return -1;

        ff::cout << "\n\n";
        ff::cout << "Executing WordCount with parameters:" << endl;
        ff::cout << "  * source/splitter : " << source_par_deg << endl;
        ff::cout << "  * counter/sink    : " << sink_par_deg << endl;
        ff::cout << "  * running time    : " << app_run_time << " (s)\n";
    //}
    
    /// application starting time
    unsigned long app_start_time = current_time_nsecs();

    std::vector<ff_node*> L;  // left and right workers of the A2A
    std::vector<ff_node*> R;

        
    ff_a2a a2a(false, qlen, qlen, true);

    auto G1 = a2a.createGroup("G1");
    auto G2 = a2a.createGroup("G2");

    for (size_t i=0;i<source_par_deg; ++i) {  
        auto* child = new ff_pipeline(false, qlen, qlen, true);
        child->add_stage(new Source(app_start_time));
        child->add_stage(new Splitter);   
        //auto* combine = new ff_comb(new Source(app_start_time), new Splitter, true, true);
        L.push_back(child);
        G1 << child;
    }

    for (size_t i=0;i<sink_par_deg; ++i) {
        auto child = new ff_pipeline(false, qlen, qlen, true);
        child->add_stage(new Counter);
        child->add_stage(new Sink);
        //auto* combine = new ff_comb(new Counter, new Sink, true, true);
        R.push_back(child);
        G1 << child;
    }


    for (size_t i=0;i<source_par_deg; ++i) {        
         auto* child = new ff_pipeline(false, qlen, qlen, true);
        child->add_stage(new Source(app_start_time));
        child->add_stage(new Splitter);   
        //auto* combine = new ff_comb(new Source(app_start_time), new Splitter, true, true);
        L.push_back(child);
        G2 << child;
    }

    for (size_t i=0;i<sink_par_deg; ++i) {
         auto child = new ff_pipeline(false, qlen, qlen, true);
        child->add_stage(new Counter);
        child->add_stage(new Sink);
        //auto* combine = new ff_comb(new Counter, new Sink, true, true);
        R.push_back(child);
        G2 << child;
    }

    a2a.add_firstset(L, 0, true);
    a2a.add_secondset(R, true);

#if 0    
    if (DFF_getMyGroup() == "G1") {
        threadMapper::instance()->setMappingList("0,1,2,3,4,5,6,7,8,9,10,11, 24,25,26,27,28,29,30,31,32,33,34,35");        
    } else {
        threadMapper::instance()->setMappingList("12,13,14,15,16,17,18,19,20,21,22,23, 36,37,38,39,40,41,42,43,44,45,46,47");
    }
#endif      

    ff::cout << "Starting \n\n";
    /// evaluate topology execution time
    volatile unsigned long start_time_main_usecs = current_time_usecs();
    if (a2a.run_and_wait_end()<0) {
        error("running pipeMain\n");
        return -1;
    }
    volatile unsigned long end_time_main_usecs = current_time_usecs();
    ff::cout << "Exiting" << endl;
    double elapsed_time_seconds = (end_time_main_usecs - start_time_main_usecs) / (1000000.0);
    ff::cout << "elapsed time     : " << elapsed_time_seconds << "(s)\n";
   

        ff::cout << "total_lines sent : " << total_lines << "\n";
        ff::cout << "total_bytes sent : " << std::setprecision(3) << total_bytes/1048576.0 << "(MB)\n";
        //double throughput = total_lines / elapsed_time_seconds;
        //double mbs = (double)((total_bytes / 1048576) / elapsed_time_seconds);
        //std::cout << "Measured throughput: " << (int) throughput << " lines/second, " << mbs << " MB/s" << endl;

   
        ff::cout << "words            : " << global_words << "\n";
        ff::cout << "unique           : " << global_unique<< "\n";

    
    return 0;
}
