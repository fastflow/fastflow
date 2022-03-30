/*
 * FastFlow concurrent network:
 *
 *                        | -> Rnode2 -> |    
 *                        |              |
 *           |-> Lnode1 ->| -> Rnode2 -> |
 *  Source ->|            |              | ---> Sink
 *           |-> Lnode2 ->| -> Rnode2 -> |
 *                        |              |
 *                        | -> Rnode2 -> |
 * 
 *
 *  distributed version:
 *
 *
 *     G1                        G2
 *   --------          -----------------------
 *  |        |        |           |-> Rnode1  |
 *  | Source | ---->  | Lnode1 -->|           | -->|     ------
 *  |        |  |     |           |-> Rnode2  |    |    |      |
 *   --------   |      -----------------------     |--> | Sink |
 *              |               |  ^               |    |      |
 *              |               |  |               |     ------
 *              |               v  |               |       G4   
 *              |      -----------------------     | 
 *               ---> |           |-> Rnode3  | -->|  
 *                    | Lnode2 -->|           |
 *                    |           |-> Rnode4  |
 *                     -----------------------
 *                               G3
 *
 *
 *  ** MANUAL ** implementation of the distributed groups!
 *
 *  WARNING: This test is intended for FastFlow developers!!
 *
 */

#include <iostream>
#include <ff/dff.hpp>
#include <ff/distributed/ff_dadapters.hpp>
#include <mutex>
#include <map>

using namespace ff;
std::mutex mtx;



struct myTask_t {
	std::string str;
	struct S_t {
		long  t;
		float f;
	} S;

	template<class Archive>
	void serialize(Archive & archive) {
		archive(str, S.t, S.f);
	}

};



struct Source : ff_monode_t<myTask_t>{
    myTask_t* svc(myTask_t*){
        for(int i = 0; i < 100; i++) {
			myTask_t* task = new myTask_t;
			task->str="Hello";
			task->S.t = i;
			task->S.f = i*1.0;
			ff_send_out(task);
		}
		return EOS;
    }
};

struct Lnode : ff_monode_t<myTask_t>{
    int numWorker, generatorID;
    Lnode(int numWorker, int generatorID) : numWorker(numWorker), generatorID(generatorID) {}

    myTask_t* svc(myTask_t* in){
        std::cout << "Lnode " << generatorID << "( " << get_my_id() << ") starting generating tasks!" << std::endl;
        for(int i = 0; i < numWorker; i++) {
			myTask_t* out = new myTask_t;
			out->str = std::string(std::string("Task" + std::to_string(i) + " generated from " + std::to_string(generatorID) + " for " + std::to_string(i)));
			out->S.t = in->S.t;
			out->S.f = in->S.f;
			
			ff_send_out_to(out, i);
        }
		delete in;
        std::cout << "Lnode " << generatorID << " generated all task sending now EOS!" << std::endl;
        return GO_ON;
    }
};


struct Rnode : ff_minode_t<myTask_t>{
    int ID;
    Rnode(int id): ID(id) {}
    myTask_t* svc(myTask_t* in){
        in->str =std::string(std::string(in->str + " received by Rnode " + std::to_string(ID) + " from channel " +  std::to_string(get_channel_id())));
        return in;
    }
};

struct Sink : ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* in){
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << in->str << std::endl;
		delete in;
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

    ff_farm gFarm;
    ff_a2a a2a;

    if (atoi(argv[1]) == 0){
        gFarm.add_collector(new ff_dsender({g1, g2}, "G0"));
        gFarm.add_workers({new WrapperOUT(new Source(), 1, true)});

        gFarm.run_and_wait_end();
        return 0;
    } else if (atoi(argv[1]) == 1){
        gFarm.add_emitter(new ff_dreceiverH(g1, 2, {{0, 0}}, {0,1}, {"G2"}));
        gFarm.add_collector(new ff_dsenderH({g2,g3}, "G1", {"G2"}));

		auto s = new Lnode(4,0);
		
        auto ea = new ff_comb(new WrapperIN(new ForwarderNode(s->deserializeF)), new EmitterAdapter(s, 4, 0, {{0,0}, {1,1}}, true), true, true);

        a2a.add_firstset<ff_node>({ea, new SquareBoxLeft({{0,0}, {1,1}})});
        auto rnode0 = new Rnode(0);
		auto rnode1 = new Rnode(1);
        a2a.add_secondset<ff_node>({
									new ff_comb(new CollectorAdapter(rnode0, {0}, true),
												new WrapperOUT(new ForwarderNode(rnode0->serializeF), 0, 1, true)),
									new ff_comb(new CollectorAdapter(rnode1, {0}, true),
												new WrapperOUT(new ForwarderNode(rnode1->serializeF), 1, 1, true)),
									new SquareBoxRight});  // this box should be the last one!

    } else if (atoi(argv[1]) == 2) {
        gFarm.add_emitter(new ff_dreceiverH(g2, 2, {{1, 0}}, {2,3}, {"G1"}));
        gFarm.add_collector(new ff_dsenderH({g1, g3}, "G2", {"G1"}));

		gFarm.cleanup_emitter();
		gFarm.cleanup_collector();

		auto s = new Lnode(4,1);                                                                           
		auto ea = new ff_comb(new WrapperIN(new ForwarderNode(s->deserializeF)), new EmitterAdapter(s, 4, 1, {{2,0}, {3,1}}, true), true, true);

        a2a.add_firstset<ff_node>({ea, new SquareBoxLeft({std::make_pair(2,0), std::make_pair(3,1)})}, 0, true);

        auto rnode2 = new Rnode(2);
		auto rnode3 = new Rnode(3);
		a2a.add_secondset<ff_node>({
									new ff_comb(new CollectorAdapter(rnode2, {1}, true),
												new WrapperOUT(new ForwarderNode(rnode2->serializeF), 2, 1, true), true, true),
									new ff_comb(new CollectorAdapter(rnode3, {1}, true),
												new WrapperOUT(new ForwarderNode(rnode3->serializeF), 3, 1, true), true, true),
									new SquareBoxRight   // this box should be the last one!
			                        }, true);		
        
    } else {
        gFarm.add_emitter(new ff_dreceiver(g3, 2));
		gFarm.add_workers({new WrapperIN(new Sink(), 1, true)});
		
		gFarm.run_and_wait_end();
		return 0;
    }
    gFarm.add_workers({&a2a});
    gFarm.run_and_wait_end();

	return 0;
}
