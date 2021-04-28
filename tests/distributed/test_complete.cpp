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
 *  /<------------ pipe1 -------------->/  /<- pipe2 ->/
 *  /<-------------------- pipeMain ------------------>/
 *
 *
 *  distributed version:
 *
 *   --------          --------
 *  |  pipe1 | ---->  |  pipe2 |
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
    ff_pipeline mainPipe;
    ff_a2a a2a;
    Source s;
    ff_pipeline sp;
	sp.add_stage(&s);
	sp.add_stage(&a2a);
    ff_pipeline sinkp;
	Sink sink;
	sinkp.add_stage(&sink);
    mainPipe.add_stage(&sp);
    mainPipe.add_stage(&sinkp);

    MoNode sx1, sx2, sx3;
    MiNode dx1, dx2, dx3;

    a2a.add_firstset<MoNode>({&sx1, &sx2, &sx3});
    a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});
	// -----------------------

	// defining the distributed groups
    dGroup g1 = sp.createGroup("G1");
    dGroup g3 = sinkp.createGroup("G2");

    g1.out << &dx1 << &dx2 << &dx3;
    g3.in << &sink;
    // ----------------------

	// running the distributed groups
    if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
	}
    
    return 0;
}
