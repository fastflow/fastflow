/*
 * FastFlow concurrent network:
 *
 *  /<------------- pipe -------------->/
 *            /<-------- a2a -------->/
 *              
 *                      | -> Node3 ->| 
 *           |-> Node2->|            |
 *  Node1--->|          | -> Node3 ->| -- 
 *    ^      |-> Node2->|            |   |
 *    |                 | -> Node3 ->|   |
 *    |                                  |
 *     ----------------------------------
 *
 *
 * distributed version:
 *
 *  Node1    -> G1
 *  a2a      -> G2
 *
 */


#include <iostream>
#include <ff/dff.hpp>

using namespace ff;

struct myTask_t {
	myTask_t() {}
	myTask_t(myTask_t* t){
		str = std::string(t->str);
		S.t = t->S.t;
		S.f = t->S.f;
	}

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

struct Node1Helper: ff_minode_t<myTask_t> {
    myTask_t* svc(myTask_t*in) {
		return in;
	}
};
		
struct Node1: ff_monode_t<myTask_t>{
	Node1(long ntasks):ntasks(ntasks),cnt(ntasks) {}
    myTask_t* svc(myTask_t*in){
		long min = std::min(cnt,appbatch);
		for(long i=0; i < min; i++) {
			myTask_t* task = new myTask_t;
			task->str="Hello";
			task->S.t = i;
			task->S.f = i*1.0;
			ff_send_out(task);
			--cnt;
		}
		if (in) {
			++back;
			delete in;
		}
		if (back == ntasks) {
			return EOS;
		}
		return GO_ON;
    }
	void svc_end() {
		if (back != ntasks) {
			std::cerr << "ERROR\n";
			exit(-1);
		}
		std::cout << "RESULT OK\n";
	}
	
	const long ntasks;
	long back=0,cnt;
	long appbatch=10;
};


struct Node2: ff_monode_t<myTask_t>{
	myTask_t* svc(myTask_t* t){
		t->str += std::string(" World");
		return t;
	}
};
struct Node3: ff_minode_t<myTask_t>{ 
    myTask_t* svc(myTask_t* t){
		t->str   = "Feedback!";
		t->S.t  += 1;
		t->S.f  += 1.0;
		//std::cout << "Node3(" << get_my_id() << ") sending task back\n";
		return t;
    }
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	long ntasks = 100;
	if (argc>1) {
		if (argc != 2) {
			std::cerr << "usage: " << argv[0]
					  << " ntasks\n";
			return -1;
		}
		ntasks = std::stol(argv[1]);
	}

	ff_pipeline pipe;
	ff_comb n1(new Node1Helper, new Node1(ntasks), true, true);

	ff_a2a a2a;
	a2a.add_firstset<ff_node>({new Node2, new Node2}, 0, true);
	a2a.add_secondset<ff_node>({new Node3, new Node3, new Node3}, true); 

#if defined(POINTER_BASED_VERSION)
	// it creates the distributed groups from level1 nodes of the skeleton tree
 
	pipe.add_stage(&n1);  // the default pointer-based add_stage 
	pipe.add_stage(&a2a); // the default pointer-based add_stage 
	pipe.wrap_around();

    //----- defining the distributed groups ------
	n1.createGroup("G1");
	a2a.createGroup("G2");
    // -------------------------------------------	
#else 
	// it creates the distributed groups from the level0 main pipeline
	
	// this version of add_stage adds the stage to the pipeline by using references to the nodes
	// it means the reference cannot used in the '<<' operator to add the node to the distributed group
	// because of the copies
	pipe.add_stage(n1);    // a copy of n1 is added to the pipeline as first stage
	pipe.add_stage(a2a);   // a copy of the a2a is added to the pipeline as second stage
	pipe.wrap_around();
	
    //----- defining the distributed groups ------
	auto G0 = pipe.createGroup("G1");
	G0 << pipe.get_firststage(); 
	auto G1 = pipe.createGroup("G2");
	G1 << pipe.get_laststage();
	// -------------------------------------------	

#endif	

	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}

	return 0;
}
