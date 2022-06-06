/* 
 * FastFlow concurrent network:
 *
 * 
 *             ------------------------
 *            |            |--> Sink1  | 
 *            |  Source1-->|           | 
 *            |            |--> Sink2  |
 *            |  Source2-->|           |
 *            |            |--> Sink3  |
 *            |  Source3-->|           |
 *            |            |--> Sink4  |
 *             ------------------------                            
 *
 *  distributed version:
 *
 *             ------------        -----------
 *            |            |      | -> Sink1  | 
 *            |  Source1-> |      |           | 
 *            |            |      | -> Sink2  |
 *            |  Source2-->| ---> |           |
 *            |            |      | -> Sink3  |
 *            |  Source3-->|      |           |
 *            |            |      | -> Sink4  |
 *             ------------        -----------
 *                G1                   G2
 *
 */


#include <map>
#include <mutex>
#include <iostream>
#include <ff/dff.hpp>

using namespace ff;
std::mutex mtx;   // used for pretty printing

struct Source : ff_monode_t<std::string>{

	std::string* svc(std::string* in){
		int numWorkers = get_num_outchannels();

        for(int i = 0; i < numWorkers; i++)
            ff_send_out_to(new std::string("Task generated from " + std::to_string(get_my_id()) + " for " + std::to_string(i)), i);
        return EOS;
    }
};


struct Sink : ff_minode_t<std::string>{
    int sinkID;
    Sink(int id): sinkID(id) {}

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << *in << " received by Sink " << sinkID << " from " << get_channel_id() << "\n";
        delete in;
        return this->GO_ON;
    }
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}

    ff_a2a a2a;
    std::vector<Source*> firstSet;
    std::vector<Sink*> secondSet;
    auto g1 = a2a.createGroup("G1");
    auto g2 = a2a.createGroup("G2");

	for(int i  = 0 ; i < 3; i++){
        auto s = new Source();
        firstSet.push_back(s);
        g1 << s;
    }

    for(int i  = 0; i < 4; i++){
        auto s = new Sink(i);
        secondSet.push_back(s);
        g2 << s;
    }

    a2a.add_firstset(firstSet);
    a2a.add_secondset(secondSet);

    ff_Pipe p(&a2a);
    if (p.run_and_wait_end()<0) {
		error("running pipe");
		return -1;
	}
	return 0;
}
