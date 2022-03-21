/* 
 * FastFlow concurrent network:
 * 
 *                          |--> MiNode 
 *             |-> MoNode-->|           
 *   MoNode -->|            |--> MiNode 
 *             |-> MoNode-->|           
 *                          |--> MiNode 
 *
 *            /<-------- a2a -------->/
 * /<----------- pipeMain ------------->/
 *
 *  distributed version:
 *                                         G2                  
 *                     G1             -----------
 *     G0         --------------     | -> MiNode |
 *   --------    | -> MoNode -> |    |           |
 *  | MoNode |-->|              | -->| -> MiNode |
 *   --------    | -> MoNode -> |    |           |
 *                --------------     | -> MiNode |
 *                                    -----------
 */


#include <ff/dff.hpp>
#include <cmath>
#include <iostream>
#include <mutex>
#include <chrono>

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
struct DataType {
	long x;
	long y;

	template<class Archive>
	void serialize(Archive & archive) {
		archive(x,y);
	}	
};

struct MoNode : ff::ff_monode_t<DataType>{
    MoNode(int itemsToGenerate):items(itemsToGenerate) {}
	
    DataType* svc(DataType* in){
		if (items) {
			for(int i=0; i< items; i++){
				auto d = new DataType;
				d->x=i, d->y=i+1;
				ff_send_out(d);
			}        
			return this->EOS;
		}
		++cnt;
		return in;
    }
	void svc_end() {
		if (items) {
			const std::lock_guard<std::mutex> lock(mtx);
			ff::cout << "[MoNode" << this->get_my_id() << "] Generated Items: " << items << std::endl;
		}
		if (cnt) {
			const std::lock_guard<std::mutex> lock(mtx);
			ff::cout << "[MoNode" << this->get_my_id() << "] Received Items: " << cnt << std::endl;
		}
		
	}

    long items, cnt=0;
};

struct MiNode : ff::ff_minode_t<DataType>{
    int processedItems = 0;
    int execTime;
	bool checkdata;
    MiNode(int execTime): execTime(execTime) {}

    DataType* svc(DataType* in){
		active_delay(this->execTime);
		++processedItems;
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

    if (argc != 4){
        std::cout << "Usage: " << argv[0] << " #items #nw_sx #nw_dx"  << std::endl;
        return -1;
    }
    int items = atoi(argv[1]);
    int numWorkerSx = atoi(argv[2]);
    int numWorkerDx = atoi(argv[3]);
	
    ff_pipeline mainPipe;
	MoNode generator(items);
	mainPipe.add_stage(&generator);
	ff_a2a a2a;
    mainPipe.add_stage(&a2a);

    std::vector<MoNode*> sxWorkers;
    std::vector<MiNode*> dxWorkers;

    for(int i = 0; i < numWorkerSx; i++)
        sxWorkers.push_back(new MoNode(0));
	
    for(int i = 0; i < numWorkerDx; i++)
        dxWorkers.push_back(new MiNode((i+1)*10));
	
    a2a.add_firstset(sxWorkers, 1);  // enabling on-demand distribution policy
    a2a.add_secondset(dxWorkers);
	
	//----- defining the distributed groups ------

	auto G0 = generator.createGroup("G0");
    auto G1 = a2a.createGroup("G1");
    auto G2 = a2a.createGroup("G2");

    for(int i = 0; i < numWorkerSx; i++) 
		G1  << sxWorkers[i];
    for(int i = 0; i < numWorkerDx; i++) 
		G2 << dxWorkers[i];

    // -------------------------------------------

	
    if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
    }
    return 0;
}
