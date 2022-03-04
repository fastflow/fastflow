/*
 *           |-> Forwarder1 ->|
 *           |                | -> |-> Sink1 ->|  
 *  Source ->|-> Forwarder2 ->|    |           | -> StringPrinter
 *           |                | -> |-> Sink2 ->|
 *           |-> Forwarder3 ->|  
 *
 * 
 *  G0: Source
 *  G1: Forwarer1
 *  G2: Sink1
 *  G3: Forwarder3, Sink2
 *  G4: StringPrinter
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
        for(int i = 0; i < numWorker; i++)
			ff_send_out_to(new std::string("Task" + std::to_string(i) + " generated from " + std::to_string(generatorID) + " for " + std::to_string(i)), i);
        
        std::cout << "Source generated all task sending now EOS!" << std::endl;
        return EOS;
    }
};


struct Sink : ff_minode_t<std::string>{
    int sinkID;
    Sink(int id): sinkID(id) {}
    std::string* svc(std::string* in){
        std::string* output = new std::string(*in + " received by Sink " + std::to_string(sinkID) + " from " +  std::to_string(get_channel_id()));
        delete in;
        return output;
    }
};

struct StringPrinter : ff_node_t<std::string>{
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Received something! Addr:" << in << "\n";
#if 1
        try {
            std::cout << *in << std::endl;
            delete in;
        } catch (const std::exception& ex){
            std::cerr << ex.what();
        }
#endif		
        return this->GO_ON;
    }
};


struct ForwarderNode : ff_node{ 
        ForwarderNode(std::function<void(void*, dataBuffer&)> f){
            this->serializeF = f;
        }
        ForwarderNode(std::function<void*(dataBuffer&)> f){
            this->deserializeF = f;
        }
        void* svc(void* input){return input;}
};


int main(int argc, char*argv[]){

    if (argc != 2){
        std::cerr << "Execute with the index of process!" << std::endl;
        return 1;
    }

    ff_endpoint g1("127.0.0.1", 8001);
    g1.groupName = "G1";

    ff_endpoint g2("127.0.0.1", 8002);
    g2.groupName = "G2";

    ff_endpoint g3("127.0.0.1", 8003);
    g3.groupName = "G3";

    ff_endpoint g4("127.0.0.1", 8004);
    g4.groupName = "G4";

    ff_farm gFarm;
    ff_a2a a2a;

    a2a.createGroup("G1");
    a2a.createGroup("G2");
    a2a.createGroup("G3");
    ff_pipeline dummypipe; dummypipe.createGroup("G4"); dummypipe.createGroup("G0"); 


    if (atoi(argv[1]) == 0){
        dGroups::Instance()->setRunningGroup("G0");
        gFarm.add_collector(new ff_dsender({g1, g3}));
        gFarm.add_workers({new WrapperOUT(new RealSource(), 1, true)});

        gFarm.run_and_wait_end();
        return 0;
    } else if (atoi(argv[1]) == 1){
        dGroups::Instance()->setRunningGroup("G1");
        gFarm.add_emitter(new ff_dreceiver(g1, 1, {{0, 0}}));
        gFarm.add_collector(new ff_dsender({g2,g3}));

        gFarm.add_workers({new WrapperINOUT(new Source(2,0), 1, true)});
        
        gFarm.run_and_wait_end();
        return 0;

    } else if (atoi(argv[1]) == 2){
        dGroups::Instance()->setRunningGroup("G2");
        gFarm.add_emitter(new ff_dreceiver(g2, 2, {{0, 0}}));
        gFarm.add_collector(new ff_dsender(g4));

        gFarm.add_workers({new WrapperINOUT(new Sink(0), 1, true)});
		
        gFarm.run_and_wait_end();
        return 0;

    } else if (atoi(argv[1]) == 3) {
        dGroups::Instance()->setRunningGroup("G3");
        gFarm.add_emitter(new ff_dreceiverH(g3, 2, {{1, 0}}, {1}));
        gFarm.add_collector(new ff_dsenderH({g2, g4}));
		gFarm.cleanup_emitter();
		gFarm.cleanup_collector();

		auto s = new Source(2,1);
		auto ea = new ff_comb(new WrapperIN(new ForwarderNode(s->deserializeF)), new EmitterAdapter(s, 2, 1, {{1,0}}, true), true, true);

        a2a.add_firstset<ff_node>({ea, new SquareBoxLeft({std::make_pair(1,0)})}, 0, true);

        auto sink = new Sink(1);
		a2a.add_secondset<ff_node>({
									new ff_comb(new CollectorAdapter(sink, {1}, true),
												new WrapperOUT(new ForwarderNode(sink->serializeF), 1, true), true, true),
									new SquareBoxRight
			                        }, true);

		
        
    } else {
        dGroups::Instance()->setRunningGroup("G4");
        gFarm.add_emitter(new ff_dreceiver(g4, 2));
         gFarm.add_workers({new WrapperIN(new StringPrinter(), 1, true)});

         gFarm.run_and_wait_end();
         return 0;
    }
    gFarm.add_workers({&a2a});
    gFarm.run_and_wait_end();
}
