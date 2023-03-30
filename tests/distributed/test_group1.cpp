/* 
 * FastFlow concurrent network:
 *
 * 
 *             -----------------------------
 *            |            |--> MiNode1 --> | 
 *            |  MoNode1-->|                | 
 *  Source -->|            |--> MiNode2 --> | ---->  Sink
 *            |  MoNode2-->|                |
 *            |            |--> MiNode3 --> |
 *             -----------------------------                            
 *            |<---------- A2A ------- ---->| 
 *  |<-------------------  pipe ----------------------->|
 *
 *
 *  distributed version:
 *
 *     G1                        G2
 *   --------          -----------------------
 *  |        |        |           |-> MiNode1 |
 *  | Source | ---->  | MoNode1 ->|           | -->|     ------
 *  |        |  |     |           |-> MiNode2 |    |    |      |
 *   --------   |      -----------------------     |--> | Sink |
 *              |               |  ^               |    |      |
 *              |               |  |               |     ------
 *              |               v  |               |       G4   
 *              |      -----------------------     | 
 *               ---> |                       | -->|  
 *                    | MoNode2 ->|-> MiNode3 |
 *                     -----------------------
 *                               G3
 */

#include <ff/dff.hpp>
#include <iostream>
#include <mutex>

#define ITEMS 100
std::mutex mtx;  // used only for pretty printing

using namespace ff;

struct Source : ff_monode_t<int>{
    int* svc(int* i){
        for(int i=0; i< ITEMS; i++)
            ff_send_out(new int(i));
        
        return EOS;
    }

    void svc_end(){
        ff::cout << "Source ended!\n"; 
    }
};

struct MoNode : ff_monode_t<int>{
    int processedItems = 0;
    int* svc(int* i){
        ++processedItems;
        return i;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "[SxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};

struct MiNode : ff_minode_t<int>{
    int processedItems = 0;
    int* svc(int* i){
        ++processedItems;
        return i;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "[DxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};

struct Sink : ff_minode_t<int>{
    int sum = 0;
    int* svc(int* i){
        sum += *i;
        delete i;
        return GO_ON;
    }

    void svc_end() {
        int local_sum = 0;
        for(int i = 0; i < ITEMS; i++) local_sum += i;
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Sum: " << sum << " (Expected: " << local_sum << ")" << std::endl;
    }
};


int main(int argc, char*argv[]){

    if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

	// defining the concurrent network
    ff_pipeline mainPipe;
    Source source;
    ff_a2a a2a;
	Sink sink;
	mainPipe.add_stage(&source);
	mainPipe.add_stage(&a2a);
	mainPipe.add_stage(&sink);

    MoNode sx1, sx2;
    MiNode dx1, dx2, dx3;

    a2a.add_firstset<MoNode>({&sx1, &sx2});
    a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});

	//----- defining the distributed groups ------
    source.createGroup("G1");
	a2a.createGroup("G2") << &sx1 << &dx1 << &dx2;
	a2a.createGroup("G3") << &sx2 << &dx3;
	sink.createGroup("G4");

    // -------------------------------------------

	// running the distributed groups
    if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
	}
    
    return 0;
}
