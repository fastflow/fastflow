#include <iostream>
#include<ff/dff.hpp>
#include<ff/distributed/ff_dadapters.hpp>
#include <mutex>
#include <map>


using namespace ff;

#define WORKERS 10

struct Source : ff_monode_t<std::string>{
    int numWorker;
    Source(int numWorker) : numWorker(numWorker) {}

    std::string* svc(std::string* in){
        for(int i = 0; i < numWorker; i++)
            ff_send_out_to(new std::string("Task generated for " + std::to_string(i) + " passed to"), i);
        return EOS;
    }
};

struct Worker : ff_node_t<std::string>{
    int numWorker;
    Worker(int numWorker) : numWorker(numWorker) {}

    std::string* svc(std::string * in){
        std::string * output = new std::string(*in + " " + std::to_string(numWorker));
        delete in;
        return output;
    }
};

struct Sink : ff_minode_t<std::string>{
    std::string* svc(std::string* in){
        std::cout << *in << " received by Collector from " << get_channel_id() << std::endl;
        delete in;
        return this->GO_ON;
    }
};


int main(int argc, char*argv[]){

    if (argc != 2){
        std::cerr << "Execute with the index of process!" << std::endl;
        return 1;
    }

    if (atoi(argv[1]) == 0){
        ff_farm f;
        f.add_emitter(new EmitterAdapter(new Source(WORKERS), WORKERS, WORKERS/2, true));
        std::vector<ff_node*> workers;
        for(int i = 0; i < WORKERS/2; i++)
            workers.push_back(new WrapperOUT<true, std::string, std::string>(new Worker(i), 1, true, FARM_GATEWAY));
        workers.push_back(new WrapperOUT<true, std::string, std::string>(new SquareBoxEmitter<std::string>(), 1, true));
        f.add_workers(workers);
        f.add_collector(new ff_dsender(ff_endpoint("127.0.0.1", 8001)));
        f.run_and_wait_end();

    } else {
        ff_farm f;
        std::map<int, int> routingTable;
        std::vector<ff_node*> workers;
        int j = 1;
        workers.push_back(new WrapperIN<true, std::string, std::string>(new SquareBoxCollector<std::string>(), 1, true));
        routingTable.emplace(FARM_GATEWAY, 0);
        for(int i = WORKERS / 2; i < WORKERS; i++){
            workers.push_back(new WrapperIN<true, std::string, std::string>(new Worker(i), 1, true));
            routingTable.emplace(i, j++);
        }

        f.add_workers(workers);
        f.add_emitter(new ff_dreceiver(ff_endpoint("127.0.0.1", 8001), 1, routingTable));
        f.add_collector(new CollectorAdapter(new Sink, 5, 5, true), true);

        f.run_and_wait_end();
    }
    /*
    ff_farm farm;
    farm.add_emitter(new Source(WORKERS));
    farm.add_collector(new Sink, true);
    std::vector<ff_node*> workers;
    for(int i = 0; i < WORKERS; i++)
        workers.push_back(new Worker(i));
    farm.add_workers(workers);

    farm.run_and_wait_end();
    */
}