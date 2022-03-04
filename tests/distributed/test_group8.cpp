/* 
 * FastFlow concurrent network:
 * 
 *                       |--> MiNode -->|
 *              MoNode-->|              |
 *  Source -->           |--> MiNode -->|---->  Sink
 *              MoNode-->|              |
 *                       |--> MiNode -->|
 *
 *             /<--------- a2a -------->/
 *  /<-------------------- pipeMain ------------------>/
 *
 *  distributed version:
 *
 *     G1                        G2                   G3
 *   --------          -----------------------      ------
 *  |        |        |           |-> MiNode1 |    |      |
 *  | Source | ---->  | MoNode1 ->|           | -->| Sink |   
 *  |        |        |           |-> MiNode2 |    |      |
 *   --------         | MoNode2 ->|           |     ------
 *                    |           |-> MiNode3 |
 *                     -----------------------
 *              
 *                               
 */

#include <ff/dff.hpp>
#include <iostream>
#include <mutex>
using namespace ff;
#define ITEMS 100
std::mutex mtx;  // used only for pretty printing

struct Source : ff::ff_monode_t<int>{
    int* svc(int* i){
        for(int i=0; i< ITEMS; i++)
            ff_send_out(new int(i));
        
        return this->EOS;
    }
};

struct MoNode : ff::ff_monode_t<int>{
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

struct MiNode : ff::ff_minode_t<int>{
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

struct Sink : ff::ff_minode_t<int>{
    int sum = 0;
    int* svc(int* i){
        sum += *i;
        delete i;
        return this->GO_ON;
    }

    void svc_end() {
        int local_sum = 0;
        for(int i = 0; i < ITEMS; i++) local_sum += i;
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << "Sum: " << sum << " (Expected: " << local_sum << ")" << std::endl;
    }
};


int main(int argc, char*argv[]){
    DFF_Init(argc, argv);
    ff_pipeline mainPipe;
    ff::ff_a2a a2a;
    Source s;
    Sink sink;
    ff_Pipe sp(s);
    ff_Pipe sinkp(sink);
    mainPipe.add_stage(&sp);
    mainPipe.add_stage(&a2a);
    mainPipe.add_stage(&sinkp);

    MoNode sx1, sx2, sx3;
    MiNode dx1, dx2, dx3;

    a2a.add_firstset<MoNode>({&sx1, &sx2, &sx3});
    a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});

	//----- defining the distributed groups ------

    auto g1 = sp.createGroup("G1");
    auto g2 = a2a.createGroup("G2");
    auto g3 = sinkp.createGroup("G3");
	
    // -------------------------------------------
	
	if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
	}

}
