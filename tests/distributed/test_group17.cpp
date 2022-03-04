/* 
 * FastFlow concurrent network:
 *
 * 
 *             -----------------------------
 *            |            |--> Sink1 --> | 
 *            |  Source1-->|              | 
 *            |            |--> Sink2 --> |
 *            |  Source2-->|              |
 *            |            |--> Sink3 --> |
 *             -----------------------------                            
 *
 *  distributed version:
 *
 *  G1: all Source(s)
 *  G2: all Sink(s)
 *
 */


#include <iostream>
#include<ff/dff.hpp>
#include <mutex>
#include <map>

using namespace ff;
std::mutex mtx;

struct Source : ff_monode_t<std::string>{
    int numWorker, generatorID;
    Source(int numWorker, int generatorID) : numWorker(numWorker), generatorID(generatorID) {}
    
    int svc_init(){
        std::cout << "Source init called!\n";

        return 0;
    }
    
    std::string* svc(std::string* in){
        std::cout << "Entering the loop of Source\n";
        for(int i = 0; i < numWorker; i++)
            ff_send_out_to(new std::string("Task generated from " + std::to_string(generatorID) + " for " + std::to_string(i)), i);
        return EOS;
    }
};


struct Sink : ff_minode_t<std::string>{
    int sinkID;
    Sink(int id): sinkID(id) {}

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
        ff::cout << *in << " received by Sink " << sinkID << " from " << get_channel_id() << ff::endl;
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
        auto s = new Source(4, i);
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
