/*
 *     Source1 ->|
 *               |    | ->Sink1
 *     Source2 ->| -> |
 *               |    | ->Sink2
 *     Source3 ->|
 *
 *  G1: Source1, Source2, Sink1
 *  G2: Source3, Sink2
 *
 */



#include <iostream>
#include <ff/dff.hpp>
#include <ff/distributed/ff_dadapters.hpp>
#include <mutex>
#include <map>

using namespace ff;
std::mutex mtx;

struct Source : ff_monode_t<std::string>{
    int numWorker, generatorID;
    Source(int numWorker, int generatorID) : numWorker(numWorker), generatorID(generatorID) {}

    std::string* svc(std::string* in){

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
        std::cout << *in << " received by Sink " << sinkID << " from " << get_channel_id() << std::endl;
        delete in;
        return this->GO_ON;
    }
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


    ff_farm gFarm;
    ff_a2a  a2a;


    // the following are just for building this example! 
	a2a.createGroup("G1");
	a2a.createGroup("G2");

    if (atoi(argv[1]) == 0){
        dGroups::Instance()->setRunningGroup("G1");
        gFarm.add_emitter(new ff_dreceiverH(g1, 1, {{0, 0}}, {0}));
        gFarm.add_collector(new ff_dsenderH(g2));

        auto ea1 = new EmitterAdapter(new Source(2,0), 2, 0, {{0,0}}, true); ea1->skipallpop(true);
        auto ea2 = new EmitterAdapter(new Source(2,1), 2, 1, {{0,0}}, true); ea2->skipallpop(true);

        a2a.add_firstset<ff_node>({ea1, ea2, new SquareBoxLeft({std::make_pair(0,0)})});
        a2a.add_secondset<ff_node>({new CollectorAdapter(new Sink(0), {0, 1}, true), new SquareBoxRight()});

    } else {
        dGroups::Instance()->setRunningGroup("G2");
        gFarm.add_emitter(new ff_dreceiverH(g2, 1, {{0, 0}}, {1}));
        gFarm.add_collector(new ff_dsenderH(g1));

        auto ea = new EmitterAdapter(new Source(2,2), 2, 2, {{1,0}}, true); ea->skipallpop(true);

        a2a.add_firstset<ff_node>({ea, new SquareBoxLeft({std::make_pair(1,0)})});
        a2a.add_secondset<ff_node>({new CollectorAdapter(new Sink(1), {2}, true), new SquareBoxRight()});
        
    }
    gFarm.add_workers({&a2a});
    gFarm.run_and_wait_end();
}
