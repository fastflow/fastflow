/*  
 *           |-> Helper->Node21 -->|    |->  Helper->Node31 -->|             
 *           |                     |    |                      |
 *   Node1-->|-> Helper->Node22 -->| -> |->  Helper->Node32 -->| -> Node4    
 *           |                     |    |                      |
 *           |-> Helper->Node23 -->|    |->  Helper->Node33 -->|
 *
 *   /<---------- a2a0 ---------->/   /<------------ a2a ------------>/
 *   /<---------------------------- pipe ---------------------------->/
 *
 * G1: Node1
 * G2: Helper->Node21 and Helper->Node22  
 * G3: Helper->Node23
 * G4: Helper->Node31
 * G5: Helper->Node32 and Helper->Node33
 * G6: Node4
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
		std::cout << "RESULT OK\n";
	}
	long ntasks;
	long processed=0;
};

struct Helper: ff_minode_t<myTask_t> {
	myTask_t* svc(myTask_t* t) { return t; }
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
	Node1  n1(ntasks);
	Helper h21, h22, h23;
	Node2  n21, n22, n23;
	Helper h31, h32, h33;
	Node3  n31, n32, n33;
	Node4  n4(ntasks);
	ff_a2a      a2a0;
	a2a0.add_firstset<Node1>({&n1});
    a2a0.add_secondset<ff_node>({new ff_comb(&h21, &n21), new ff_comb(&h22, &n22), new ff_comb(&h23, &n23)});
	ff_a2a      a2a1;
	a2a1.add_firstset<ff_node>({new ff_comb(&h31, &n31), new ff_comb(&h32, &n32), new ff_comb(&h33, &n33)});
    a2a1.add_secondset<Node4>({&n4});
	pipe.add_stage(&a2a0);
	pipe.add_stage(&a2a1);
	
    auto G1 = a2a0.createGroup("G1");
	auto G2 = a2a0.createGroup("G2");
	auto G3 = a2a0.createGroup("G3");
    auto G4 = a2a1.createGroup("G4");
	auto G5 = a2a1.createGroup("G5");
	auto G6 = a2a1.createGroup("G6");

	/* ----------------- */ G1.out << &n1;	
    G2.in  << &h21 << &h22; G2.out << &n21 << &n22;
    G3.in  << &h23;         G3.out << &n23;
	G4.in  << &h31;         G4.out << &n31;
	G5.in  << &h32 << &h33; G5.out << &n32 << &n33;
	G6.in  << &n4;          /* ------------------- */
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
