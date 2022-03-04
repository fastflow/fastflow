/* 
 * FastFlow concurrent network:
 * 
 *           |---> Node2 ---> | -> Node3 -->|             
 *           |                |             | --> Node4 ->|
 *   Node1-->|---> Node2 ---> | -> Node3 -->|             |-> Node5
 * (emitter) |                |             | --> Node4 ->|
 *           |---> Node2 ---> | -> Node3 -->|  
 *                                
 *
 *  /<-- farm (no collector)-->//<------- a2a ----------->/
 *  /<---------------------- pipe ----------------------->/
 *
 *  NOTE: Node2 is just a standard node, not a multi-output node, 
 *        Node3 is just a ff_monode, not also a multi-input node, 
 *        therefore, there are direct connections between Node2   
 *        and Node3 nodes and not a shuffle connection among them.
 *
 *
 *  distributed version:
 *
 *  -------------------------      ------------------------
 * |         |--> Node2 ---> |    | -> Node3-->|           | 
 * |         |               |    |            | --> Node4 |    -------
 * | Node1-->|--> Node2 ---> | -->| -> Node3-->|           |-->| Node5 |
 * |         |               |    |            | --> Node4 |    -------
 * |         |--> Node2 ---> |    | -> Node3-->|           |
 *  -------------------------      ------------------------
 *           G1                                  G2                G3
 * 
 * vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv
 * REMEMBER: CHECK THAT Node2_i CAN SEND DATA ONLY TO Node3_i (same index)!!!!
 * ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
 *
 */


#include <iostream>
#include <ff/dff.hpp>

using namespace ff;

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

struct Node1: ff_monode_t<myTask_t>{
	Node1(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t*){
        for(long i=0; i< ntasks; i++) {
			myTask_t* task = new myTask_t;
			task->str="Hello";
			task->S.t = i;
			task->S.f = i*1.0;
            ff_send_out(task);
		}        
        return EOS;
    }
	const long ntasks;
};

struct Node2: ff_node_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		t->str += std::string(" World") + std::to_string(get_my_id());
        return t;
    }
};

struct Node3: ff_monode_t<myTask_t>{ 
    myTask_t* svc(myTask_t* t){
		t->S.t  += get_my_id();
		t->S.f  += get_my_id()*1.0;
        return t;
    }
};

struct Node4: ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		return t;
    }
};
struct Node5: ff_minode_t<myTask_t>{
	Node5(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t* t){
		//std::cout << "Node5: from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n" << std::flush;
		++processed;
		delete t;
        return GO_ON;
    }
	void svc_end() {
		if (processed != ntasks) {
			abort();
		}
		//ff::
			std::cout << "RESULT OK\n";
	}
	long ntasks;
	long processed=0;
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}

	long ntasks = 1000;
	if (argc>1) {
		ntasks = std::stol(argv[1]);
	}

	ff_farm farm;
	farm.add_emitter(new Node1(ntasks));
	farm.add_workers({new Node2, new Node2, new Node2});
	farm.cleanup_emitter();
	farm.cleanup_workers();
	
	ff_a2a      a2a;
	a2a.add_firstset<Node3>({new Node3, new Node3, new Node3}, 0, true);
    a2a.add_secondset<Node4>({new Node4, new Node4}, true);

	Node5 n5(ntasks);
	
	ff_pipeline pipe;
	pipe.add_stage(&farm);
	pipe.add_stage(&a2a);
	pipe.add_stage(&n5);

    //----- defining the distributed groups ------

	auto G1 = farm.createGroup("G1");
    auto G2 = a2a.createGroup("G2");
	auto G3 = n5.createGroup("G3");
	
    // -------------------------------------------

	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
