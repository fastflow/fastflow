/*
 *           |-> Forwarder1 ->|
 *           |                | -> |-> Sink1 ->|  
 *  Source ->|-> Forwarder2 ->|    |           | -> StringPrinter
 *           |                | -> |-> Sink2 ->|
 *           |-> Forwarder3 ->|  
 *
 * 
 *  G0: Source
 *  G1: Forwarer1, Forwarder2, Sink1
 *  G2: Forwarder3, Sink2
 *  G3: StringPrinter
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
        ForwarderNode(std::function<bool(void*, dataBuffer&)> f){
            this->serializeF = f;
        }
    	ForwarderNode(std::function<void*(dataBuffer&,bool&)> f){
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

    ff_farm gFarm;
    ff_a2a a2a;

    if (atoi(argv[1]) == 0){
        gFarm.add_collector(new ff_dsender({g1, g2}, "G0"));
        gFarm.add_workers({new WrapperOUT(new RealSource(), 1, true)});

        gFarm.run_and_wait_end();
        return 0;
    } else if (atoi(argv[1]) == 1){
        gFarm.add_emitter(new ff_dreceiverH(g1, 2, {{0, 0}}, {0}, {"G2"}));
        gFarm.add_collector(new ff_dsenderH({g2,g3}, "G1", {"G2"}));

		auto s = new Source(2,0);
        auto ea = new ff_comb(new WrapperIN(new ForwarderNode(s->deserializeF)), new EmitterAdapter(s, 2, 0, {{0,0}}, true), true, true);

        a2a.add_firstset<ff_node>({ea, new SquareBoxLeft({std::make_pair(0,0)})});
        auto sink = new Sink(0);
        a2a.add_secondset<ff_node>({new ff_comb(new CollectorAdapter(sink, {0}, true), new WrapperOUT(new ForwarderNode(sink->serializeF), 1, true)), new SquareBoxRight});

    } else if (atoi(argv[1]) == 2) {
        gFarm.add_emitter(new ff_dreceiverH(g2, 2, {{1, 0}}, {1}, {"G1"}));
        gFarm.add_collector(new ff_dsenderH({g1, g3}, "G2", {"G1"}));
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
        gFarm.add_emitter(new ff_dreceiver(g3, 2));
         gFarm.add_workers({new WrapperIN(new StringPrinter(), 1, true)});

         gFarm.run_and_wait_end();
         return 0;
    }
    gFarm.add_workers({&a2a});
    gFarm.run_and_wait_end();
}
