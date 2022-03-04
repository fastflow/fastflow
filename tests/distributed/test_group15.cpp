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
 *  /<------------- pipe -------------->/ 
 *  /<-------------------- pipeMain ------------------>/
 *
 *
 *  distributed version:
 *
 *   --------          --------
 *  |  pipe  | ---->  |  Sink  |
 *  |        |        |        |
 *   --------          --------
 *     G1                 G2
 *
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
};

struct MoNode : ff_monode_t<int>{
    int processedItems = 0;
    int* svc(int* i){
        ++processedItems;
        return i;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[SxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
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
        std::cout << "[DxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
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
        std::cout << "Sum: " << sum << " (Expected: " << local_sum << ")" << std::endl;
    }
};


int main(int argc, char*argv[]){
    
    if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

	// defining the concurrent network
    ff_pipeline pipe;
    Source s;
    ff_a2a a2a;
    MoNode sx1, sx2, sx3;
    MiNode dx1, dx2, dx3;
    a2a.add_firstset<MoNode>({&sx1, &sx2, &sx3});
    a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});
	pipe.add_stage(&s);
	pipe.add_stage(&a2a);
	
	Sink sink;
	ff_pipeline mainPipe;
	
	mainPipe.add_stage(&pipe);
    mainPipe.add_stage(&sink);

	//----- defining the distributed groups ------
	
    auto g1 = pipe.createGroup("G1");
    auto g2 = sink.createGroup("G2");

    // -------------------------------------------
	
	// running the distributed groups
    if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
	}
    
    return 0;
}
