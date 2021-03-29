/*  
 *                           |-> Node31 -->|             
 *             |-> Node21 -->|             |
 *   Node1---->|             |-> Node32 -->| ----> Node4    
 *             |-> Node22 -->|             |
 *                           |-> Node33 -->|
 *
 *   /<-pipe0->//<--------- a2a0 ---------->//<- pipe1->/  
 *   /<-------------..------ pipe --------------------->/
 *
 * G1: pipe0
 * G2: a2a0
 * G3: pipe1
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

struct Node2: ff_monode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		t->str += std::string(" World") + std::to_string(get_my_id());
        return t;
    }
};

struct Node3: ff_minode_t<myTask_t>{ 
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

int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	long ntasks = 1000;
	if (argc>1) {
		ntasks = std::stol(argv[1]);
	}
		
    ff_pipeline pipe, pipe0, pipe1;
	Node1  n1(ntasks);
	Node2  n21, n22;
	Node3  n31, n32, n33;
	Node4  n4(ntasks);
	pipe0.add_stage(&n1);
	pipe1.add_stage(&n4);
	ff_a2a      a2a;
	a2a.add_firstset<Node2>({&n21, &n22});
    a2a.add_secondset<Node3>({&n31, &n32, &n33});

	pipe.add_stage(&pipe0);
	pipe.add_stage(&a2a);
	pipe.add_stage(&pipe1);
	
    auto G1 = pipe0.createGroup("G1");
	auto G2 = a2a.createGroup("G2");
	auto G3 = pipe1.createGroup("G3");


	/* ----------------- */ G1.out << &n1;	
    G2.in  << &n21 << &n22; G2.out << &n31 << &n32 << &n33;
	G3.in  << &n4;          /* -------------------------- */
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
