/* 
 * FastFlow concurrent network:
 * 
 *             |--> Sink1 
 *   Source1-->|           
 *             |--> Sink2 
 *   Source2-->|           
 *             |--> Sink3
 *
 *
 *  distributed version:
 *
 *  G1: all Source(s)
 *  G2: all Sink(s)
 *
 */



#include <ff/dff.hpp>
#include <mutex>
#include <iostream>
#include <cmath>

using namespace ff;

// ------------------------------------------------------
std::mutex mtx;  // used only for pretty printing
static inline float active_delay(int msecs) {
  // read current time
  float x = 1.25f;
  auto start = std::chrono::high_resolution_clock::now();
  auto end   = false;
  while(!end) {
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    x *= sin(x) / atan(x) * tanh(x) * sqrt(x);
    if(msec>=msecs)
      end = true;
  }
  return x;
}
// -----------------------------------------------------

struct myTask_t {
	std::string str;
	struct S_t {
		long  t;
		float f;
	} S;

	template<class Archive>
	void serialize(Archive & archive) {
		archive(str, S.t, S.f);
	}
};

struct Sink: ff::ff_minode_t<myTask_t>{
	Sink(long sleep_):sleep_(sleep_) {}
    myTask_t* svc(myTask_t* t){
        active_delay(sleep_);
		++processed;
		delete t;
        return GO_ON;
    }
	void svc_end() {
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << "Node("<< sleep_ <<") Received " << processed << " tasks\n";
	}
	long sleep_;
	long processed=0;
};

struct Source: ff::ff_monode_t<myTask_t>{
	Source(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t*){
        for(long i=0; i< ntasks; i++) {
			myTask_t* task = new myTask_t;
			task->str="Hello";
			task->S.t = i;
			task->S.f = i*1.0;
            ff_send_out(task);
		}        
        return EOS;
    }
	const long ntasks;
};


int main(int argc, char*argv[]){
    DFF_Init(argc, argv);

    if (argc != 5){
        std::cout << "Usage: " << argv[0] << " #items #nw_sx #nw_dx #async"  << std::endl;
        return -1;
    }

	
    int items = atoi(argv[1]);
    int numWorkerSx = atoi(argv[2]);
    int numWorkerDx = atoi(argv[3]);
	int asyncdegree = atoi(argv[4]);

	if (numWorkerSx <= 0 ||
		numWorkerDx <= 0 ||
		asyncdegree  <= 0) {
		error("Bad parameter values\n");
		return -1;
	}
	
    ff::ff_a2a a2a;

	auto g1 = a2a.createGroup("G1");
    auto g2 = a2a.createGroup("G2");

	std::vector<Source*> sx;
	std::vector<Sink*>   dx;

	for(int i = 0; i < numWorkerSx; i++) {
		sx.push_back(new Source(items));
		g1 << sx[i];
    }
    for(int i = 0; i < numWorkerDx; i++){
		dx.push_back(new Sink((long)100*(i+1)));
		g2 << dx[i];
	}

	// enabling on-demand distribution policy with #asyncdegree buffer slots
	a2a.add_firstset<Source>(sx, asyncdegree); 
    a2a.add_secondset(dx);

    if (a2a.run_and_wait_end()<0) {
		error("running a2a");
		return -1;
	}

	return 0;
}
