/*  
 * FastFlow concurrent network:
 *                      
 *   Node1 --> Node2 --> Node3 --> Node4 
 *
 *   /<------------- pipe ------------>/
 *
 * distributed version:
 * 
 *          G1                        G2
 *   -----------------        -----------------
 *  | Node1 --> Node2 | ---> | Node3 --> Node4 |
 *  |                 |      |                 |
 *   -----------------        -----------------
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
		t->str += std::string(" World");		
		//std::cout << "Node2: " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";

        return t;
    }
};

struct Node3: ff_monode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		t->S.t  += 1;
		t->S.f  += 1.0;
        return t;
    }
};

struct Node4: ff_minode_t<myTask_t>{
	Node4(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t* t){
		//std::cout << "Node4: from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";
		++processed;
		delete t;
        return GO_ON;
    }
	void svc_end() {
		if (processed != ntasks) {
			abort();
		}
		ff::cout << "RESULT OK\n";
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

    ff_pipeline pipe;
	Node1 n1(ntasks);
	Node2 n2;
	Node3 n3;
	Node4 n4(ntasks);
	
	pipe.add_stage(&n1);
	pipe.add_stage(&n2);
	pipe.add_stage(&n3);
	pipe.add_stage(&n4);
	
    //----- defining the distributed groups ------
	
	pipe.createGroup("G1") << &n1 << &n2;
	pipe.createGroup("G2") << &n3 << &n4;
	
    // -------------------------------------------
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
