/* 
 * FastFlow concurrent network:
 *
 *      ------------------------         
 *     |  Source1-->|           |
 *     |            | --> Sink1 |
 *     |  Source2-->|           |
 *     |            | --> Sink2 |
 *     |  Source3-->|           |
 *      ------------------------ 
 *
 *  distributed version:
 *
 *    -------------------        --------------------
 *   |                   |     |                     |
 *   | Source1 -->|      |     |           |-> Sink1 |
 *   |            | -->  | --> |           |         |
 *   | Source2 -->|      |     | Source3 ->|-> Sink2 |    
 *   |                   |     |                     |
 *    -------------------       ---------------------
 *              G1                        G2
 *             
 */

#include <ff/dff.hpp>
#include <mutex>
#include <iostream>

using namespace ff;
std::mutex mtx;

struct Source : ff_monode_t<std::string>{

    std::string* svc(std::string* in){
		long outchannels = get_num_outchannels();
		ff::cout << "Expected out channels: " << outchannels << std::endl;
        for(long i = 0; i < outchannels; i++)
            ff_send_out_to(new std::string("Task generated from " + std::to_string(get_my_id()) + " for " + std::to_string(i)), i);
        
        return EOS;
    }
};


struct Sink : ff_minode_t<std::string>{
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << *in << " received by Sink " << get_my_id() << " from " << get_channel_id() << std::endl;
        delete in;
        return this->GO_ON;
    }
};

int main(int argc, char*argv[]){

	if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

	
    ff_a2a  a2a;
	Source  source1;
	Source  source2;
	Source  source3;
	Sink    sink1;
	Sink    sink2;
    a2a.add_firstset<Source>({&source1, &source2, &source3});
    a2a.add_secondset<Sink>({&sink1, &sink2});

	//----- defining the distributed groups ------

	auto g1 = a2a.createGroup("G1");
	auto g2 = a2a.createGroup("G2");

	g1 << &source1 << &source2; 
	g2 << &source3 << &sink1 << &sink2; 
	
    // -------------------------------------------

	// running the distributed groups
    if (a2a.run_and_wait_end()<0) {
		error("running a2a\n");
		return -1;
	}
	return 0;
}
