/* 
 * FastFlow concurrent network:
 * 
 *           |--> MiNode 
 *  MoNode-->|           
 *           |--> MiNode 
 *  MoNode-->|           
 *           |--> MiNode 
 *
 * /<------- a2a ------>/
 *
 * S: all left-hand side nodes
 * D: all righ-hand side nodes
 *
 */


#include <ff/dff.hpp>
#include <iostream>
#include <mutex>
#include <chrono>

//#define MANUAL_SERIALIZATION 1

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
// this assert will not be removed by -DNDEBUG
#define myassert(c) {													      \
		if (!(c)) {														\
			std::cerr << "ERROR: assert at line " << __LINE__ << " failed\n"; \
			abort();													      \
		}																      \
	}
// -----------------------------------------------------
struct ExcType {
	ExcType():contiguous(false) {}
	ExcType(bool): contiguous(true) {}
	~ExcType() {
		if (!contiguous)
			delete [] C;
	}
	
	size_t clen = 0;
	char*  C    = nullptr;
	bool contiguous;
	

	template<class Archive>
	void serialize(Archive & archive) {
	  archive(clen);
	  if (!C) {
		  myassert(!contiguous);
		  C = new char[clen];
	  }
	  archive(cereal::binary_data(C, clen));
	}
	
};


#ifdef MANUAL_SERIALIZATION
template<typename Buffer>
void serialize(Buffer&b, ExcType* input){
	b = {(char*)input, input->clen+sizeof(ExcType)};
}

template<typename Buffer>
void deserialize(const Buffer&b, ExcType*& Ptr){
    ExcType* p = new (b.first) ExcType(true);
	p->clen = b.second - sizeof(ExcType);
	Ptr = p;
}
#endif

static ExcType* allocateExcType(size_t size, bool setdata=false) {
	char* _p = (char*)malloc(size+sizeof(ExcType));	
	ExcType* p = new (_p) ExcType(true);  // contiguous allocation
	
	p->clen    = size;
	p->C       = (char*)p+sizeof(ExcType);
	if (setdata) {
		bzero(p->C, p->clen);
		p->C[0]       = 'c';
		if (size>10) 
			p->C[10]  = 'i';
		if (size>100)
			p->C[100] = 'a';
		if (size>500)
			p->C[500] = 'o';		
	}
	p->C[p->clen-1] = 'F';
	return p;
}


struct MoNode : ff::ff_monode_t<ExcType>{
    int items, execTime;
    long dataLength;
	bool checkdata;
    MoNode(int itemsToGenerate, int execTime, long dataLength, bool checkdata):
		items(itemsToGenerate), execTime(execTime), dataLength(dataLength), checkdata(checkdata) {}

    ExcType* svc(ExcType*){
       for(int i=0; i< items; i++){
		   if (execTime) active_delay(this->execTime);
		   ff_send_out(allocateExcType(dataLength, checkdata));
       }        
	   return this->EOS;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "[MoNode" << this->get_my_id() << "] Generated Items: " << items << ff::endl;
    }
};

struct MiNode : ff::ff_minode_t<ExcType>{
    int processedItems = 0;
    int execTime;
	bool checkdata;
    MiNode(int execTime, bool checkdata=false): execTime(execTime),checkdata(checkdata) {}

    ExcType* svc(ExcType* in){
      if (execTime) active_delay(this->execTime);
      ++processedItems;
	  if (checkdata) {
		  myassert(in->C[0]     == 'c');
		  if (in->clen>10) 
			  myassert(in->C[10]  == 'i');
		  if (in->clen>100)
			  myassert(in->C[100] == 'a');
		  if (in->clen>500)
			  myassert(in->C[500] == 'o');
		  ff::cout << "MiNode" << get_my_id() << " input data " << processedItems << " OK\n";
	  }
	  myassert(in->C[in->clen-1] == 'F');
	  delete in;
      return this->GO_ON;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "[MiNode" << this->get_my_id() << "] Processed Items: " << processedItems << ff::endl;
    }
};

int main(int argc, char*argv[]){
    
    if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

    if (argc < 8){
        std::cout << "Usage: " << argv[0] << " #items #byteXitem #execTimeSource #execTimeSink #np_sx #np_dx #nwXp"  << std::endl;
        return -1;
    }
	bool check = false;
    int items = atoi(argv[1]);
    long bytexItem = atol(argv[2]);
    int execTimeSource = atoi(argv[3]);
    int execTimeSink = atoi(argv[4]);
    int numProcSx = atoi(argv[5]);
    int numProcDx = atoi(argv[6]);
	int numWorkerXProcess = atoi(argv[7]);
	char* p=nullptr;
	if ((p=getenv("CHECK_DATA"))!=nullptr) check=true;
	printf("chackdata = %s\n", p);
	
    ff::ff_a2a a2a;

    std::vector<MoNode*> sxWorkers;
    std::vector<MiNode*> dxWorkers;

    for(int i = 0; i < (numProcSx*numWorkerXProcess); i++)
        sxWorkers.push_back(new MoNode(ceil((double)items/(numProcSx*numWorkerXProcess)), execTimeSource, bytexItem, check));

    for(int i = 0; i < (numProcDx*numWorkerXProcess); i++)
        dxWorkers.push_back(new MiNode(execTimeSink, check));

    a2a.add_firstset(sxWorkers);
    a2a.add_secondset(dxWorkers);

	for(int i = 0; i < numProcSx; i++){
		auto g = a2a.createGroup(std::string("S")+std::to_string(i));
		for(int j = i*numWorkerXProcess; j < (i+1)*numWorkerXProcess; j++){
			g << sxWorkers[j];
		}
	}

	for(int i = 0; i < numProcDx; i++){
		auto g = a2a.createGroup(std::string("D")+std::to_string(i));
		for(int j = i*numWorkerXProcess; j < (i+1)*numWorkerXProcess; j++){
			g << dxWorkers[j];	
		}
	}
    
    if (a2a.run_and_wait_end()<0) {
      error("running mainPipe\n");
      return -1;
    }
    return 0;
}
