/*  
 *            
 *                    | ->  Node2 -->| 
 *        Node1--> -->| ->  Node2 -->| -> Node3
 *                    | ->  Node2 -->|
 *                   /<--------- a2a -------->/
 *   /<----------------- pipe ---------------->/
 *
 *  G1: Node1 
 *  G2: all Node2 
 *  G3: Node3
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
        ff::cout << "Source exiting!" << std::endl;
        return EOS;
    }
	const long ntasks;
};

struct Node2: ff_node_t<myTask_t>{ 
    myTask_t* svc(myTask_t* t){
		t->S.t  += get_my_id();
		t->S.f  += get_my_id()*1.0;
        return t;
    }
	void eosnotify(ssize_t) {
		printf("Node3 %ld EOS RECEIVED\n", get_my_id());
		fflush(NULL);
	}
	
};

struct Node3: ff_minode_t<myTask_t>{
	Node3(long ntasks):ntasks(ntasks) {}
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
	Node2 n21, n22, n23;
	Node3 n3(ntasks);
	ff_a2a      a2a;
	a2a.add_firstset<Node2>({&n21, &n22, &n23});
    a2a.add_secondset<Node3>({&n3});
	pipe.add_stage(&n1);
	pipe.add_stage(&a2a);

	//----- defining the distributed groups ------
		
    auto G1 = n1.createGroup("G1");
    auto G2 = a2a.createGroup("G2");
    auto G3 = a2a.createGroup("G3");
	
    G2  << &n21 << &n22 << &n23;
    G3  << &n3;

	// -------------------------------------------

	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
