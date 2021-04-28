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
 * /<---- pipeMain ---->/
 */


#include <ff/dff.hpp>
#include <iostream>
#include <mutex>
#include <chrono>

#define MANUAL_SERIALIZATION 1

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
        std::cout << "[MoNode" << this->get_my_id() << "] Generated Items: " << items << std::endl;
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
		  std::cout << "MiNode" << get_my_id() << " input data " << processedItems << " OK\n";
	  }
	  myassert(in->C[in->clen-1] == 'F');
	  delete in;
      return this->GO_ON;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[MiNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};

int main(int argc, char*argv[]){
    
    DFF_Init(argc, argv);

    if (argc < 7){
        std::cout << "Usage: " << argv[0] << " #items #byteXitem #execTimeSource #execTimeSink #nw_sx #nw_dx"  << std::endl;
        return -1;
    }
	bool check = false;
    int items = atoi(argv[1]);
    long bytexItem = atol(argv[2]);
    int execTimeSource = atoi(argv[3]);
    int execTimeSink = atoi(argv[4]);
    int numWorkerSx = atoi(argv[5]);
    int numWorkerDx = atoi(argv[6]);

	char* p=nullptr;
	if ((p=getenv("CHECK_DATA"))!=nullptr) check=true;
	printf("chackdata = %s\n", p);
	
    ff_pipeline mainPipe;
    ff::ff_a2a a2a;

    mainPipe.add_stage(&a2a);

    std::vector<MoNode*> sxWorkers;
    std::vector<MiNode*> dxWorkers;

    for(int i = 0; i < numWorkerSx; i++)
        sxWorkers.push_back(new MoNode(ceil((double)items/numWorkerSx), execTimeSource, bytexItem, check));

    for(int i = 0; i < numWorkerDx; i++)
        dxWorkers.push_back(new MiNode(execTimeSink, check));

    a2a.add_firstset(sxWorkers);
    a2a.add_secondset(dxWorkers);

    //mainPipe.run_and_wait_end();

    auto g1 = a2a.createGroup("G1");
    auto g2 = a2a.createGroup("G2");


    for(int i = 0; i < numWorkerSx; i++) {
#if defined(MANUAL_SERIALIZATION)				
		g1.out <<= packup(sxWorkers[i], [](ExcType* in) -> std::pair<char*,size_t> {return std::make_pair((char*)in, in->clen+sizeof(ExcType)); });
#else
		g1.out << sxWorkers[i];
#endif
    }
    for(int i = 0; i < numWorkerDx; i++) {
#if defined(MANUAL_SERIALIZATION)						
		g2.in  <<= packup(dxWorkers[i], [](char* in, size_t len) -> ExcType* {
											ExcType* p = new (in) ExcType(true);
											p->C = in + sizeof(ExcType);
											p->clen = len - sizeof(ExcType);
											return p;
										});
#else
		g2.in << dxWorkers[i];
#endif		
    }
    
    if (mainPipe.run_and_wait_end()<0) {
      error("running mainPipe\n");
      return -1;
    }
    return 0;
}
