/*
 *           |-> Forwarder1 ->|
 *           |                | -> |-> Sink1   
 *  Source ->|-> Forwarder2 ->|    |
 *           |                | -> |-> Sink2
 *           |-> Forwarder3 ->|  
 *
 * 
 *  G0: Source
 *  G1: Forwarer1, Forwarder2, Sink1
 *  G2: Forwarder3, Sink2
 *
 */

#include <iostream>
#include <ff/dff.hpp>
#include <ff/distributed/ff_dadapters.hpp>
#include <mutex>
#include <map>

using namespace ff;
std::mutex mtx;

struct RealSource : ff_monode_t<std::string>{
    std::string* svc(std::string*){
        for(int i = 0; i < 2; i++) ff_send_out_to(new std::string("Trigger string!"), i);
        return EOS;
    }
};

struct Source : ff_monode_t<std::string>{
    int numWorker, generatorID;
    Source(int numWorker, int generatorID) : numWorker(numWorker), generatorID(generatorID) {}

    std::string* svc(std::string* in){
        delete in;
        std::cout << "Source starting generating tasks!" << std::endl;
        for(int i = 0; i < 10; i++)
			ff_send_out_to(new std::string("Task" + std::to_string(i) + " generated from " + std::to_string(generatorID) + " for " + std::to_string(i%numWorker)), i%numWorker);
        
        std::cout << "Source generated all task sending now EOS!" << std::endl;
        return EOS;
    }
};


struct Sink : ff_minode_t<std::string>{
    int sinkID;
    Sink(int id): sinkID(id) {}
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << *in << " received by Sink " << sinkID << " from " << get_channel_id() << std::endl;
        delete in;
        return this->GO_ON;
    }
};


struct ForwarderNode : ff_node{ 
	void* svc(void* input){return input;}
};


int main(int argc, char*argv[]){

    if (argc != 2){
        std::cerr << "Execute with the index of process!" << std::endl;
        return 1;
    }

    ff_farm gFarm;
    ff_a2a a2a;
    if (atoi(argv[1]) == 0){
        gFarm.add_collector(new ff_dsender({ff_endpoint("127.0.0.1", 8001, ConnectionType::EXTERNAL), ff_endpoint("127.0.0.1", 8002, ConnectionType::EXTERNAL)}));
         gFarm.add_workers({new WrapperOUT<true, std::string>(new RealSource(), 1, true, -1)});

         gFarm.run_and_wait_end();
         return 0;
    } else if (atoi(argv[1]) == 1){
        gFarm.add_emitter(new ff_dreceiver(ff_endpoint("127.0.0.1", 8001), 1, 1, {{0, 0}, {-100, 1}}));
        gFarm.add_collector(new ff_dsender(ff_endpoint("127.0.0.1", 8002, ConnectionType::INTERNAL), 1));

        //auto ea = new EmitterAdapter(new Source(2,0), 2, {{0,0}}, true); //ea->skipallpop(true);
		auto ea = new ff_comb(new WrapperIN<true,std::string>(new ForwarderNode), new EmitterAdapter(new Source(2,0), 2, {{0,0}}, true), true, true);

		
        a2a.add_firstset<ff_node>({ea, new ff_comb(new WrapperINCustom<true, std::string>(), new SquareBoxCollector<std::string>({std::make_pair(0,0)}), true, true)});
        a2a.add_secondset<ff_node>({new CollectorAdapter(new Sink(0), {0}, true), new ff_comb(new SquareBoxEmitter<std::string>({0}), new WrapperOUTCustom<true, std::string>(), true, true)});

    } else {
        gFarm.add_emitter(new ff_dreceiver(ff_endpoint("127.0.0.1", 8002), 1, 1, {{1, 0}, {-101, 1}}));
        gFarm.add_collector(new ff_dsender(ff_endpoint("127.0.0.1", 8001, ConnectionType::INTERNAL), 1));

        //auto ea = new EmitterAdapter(new Source(2,1), 2, {{1,0}}, true); //ea->skipallpop(true);
		auto ea = new ff_comb(new WrapperIN<true,std::string>(new ForwarderNode), new EmitterAdapter(new Source(2,1), 2, {{1,0}}, true), true, true);

        a2a.add_firstset<ff_node>({ea, new ff_comb(new WrapperINCustom<true, std::string>(), new SquareBoxCollector<std::string>({std::make_pair(1,0)}), true, true)});
        a2a.add_secondset<ff_node>({new CollectorAdapter(new Sink(1), {1}, true), new ff_comb(new SquareBoxEmitter<std::string>({1}), new WrapperOUTCustom<true, std::string>(),true, true)});
        
    }
    gFarm.add_workers({&a2a});
    gFarm.run_and_wait_end();
}
