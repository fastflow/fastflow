#include <ff/dff.hpp>
#include <mutex>
#include <iostream>
#include <cmath>

std::mutex mtx; 

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

    ff::ff_pipeline mainPipe;
    ff::ff_a2a a2a;

    auto g2 = a2a.createGroup("G2");

    Source s(10);
    a2a.add_firstset<Source>({&s}, 1);
    a2a.createGroup("G1").out << &s;

    std::vector<ff::ff_node*> sinks;
    for(int i = 0; i < 3; i++){
        Sink * sink = new Sink((long)1000*(i+1));
        sinks.push_back(sink);
        g2.in << sink;
    }

    a2a.add_secondset(sinks);

    mainPipe.add_stage(&a2a);
    mainPipe.run_and_wait_end();

}