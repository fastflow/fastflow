/*
 * FastFlow concurrent network:
 *
 *                   -------------
 *              ----|-------------|----
 *             |    v             |    |
 *  Node1 ---> |  Node2  ----> Node3 ->| --> Node4 
 *             |         pipe1         |      
 *              -----------------------       
 *    
 *
 *  /<------------------ pipe ------------------>/
 *
 * distributed version:
 *
 * G1: Node1
 * G2: Node2
 * G3: Node3
 * G4: Node4
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

struct Node1: ff_monode_t<myTask_t>{
	Node1(long ntasks):ntasks(ntasks) {}
	int svc_init(){
		ff::cout << "Node1 initialized!\n";
		return 0;
	}
    myTask_t* svc(myTask_t*in){
		for(long i=0; i < ntasks; i++) {
			myTask_t* task = new myTask_t;
			task->str="Hello";
			task->S.t = i;
			task->S.f = i*1.0;
			ff_send_out_to(task, 0);
		}
		return EOS;
    }
	
	const long ntasks;
};


struct Node2: ff_minode_t<myTask_t>{
	Node2(long ntasks):ntasks(ntasks) {}
	int svc_init(){
		ff::cout << "Node2 initialized!\n";
		return 0;
	}
	myTask_t* svc(myTask_t* t){
		if (fromInput()) {
			t->str += std::string(" World");
			ff::cout << "Received input task #" << ++frominputtaks << std::endl;
			return t;
		}
		ff::cout << "Received feedback task #" << ++feedbacktasks << " missing " << --ntasks <<  std::endl;
		// --ntasks;
		ff_send_out(t);
		if (ntasks == 0 && eosreceived) return EOS;
		return GO_ON;
    }
	void eosnotify(ssize_t) {
		ff::cout << "[Node2] EOS NOTIFY CALLED\n";
		eosreceived=true;
		if (ntasks==0) {
			ff_send_out(EOS);
			ntasks=-1;
		}
	}
	long ntasks, frominputtaks = 0, feedbacktasks = 0;
	bool eosreceived=false;
};
struct Node3: ff_monode_t<myTask_t>{ 
	int svc_init(){
		ff::cout << "Node3 initialized!\n";
		return 0;
	}
    myTask_t* svc(myTask_t* t){
		if (t->str == "Hello World") {
			ff::cout << "Received input task #" << ++frominputtaks << std::endl;
			t->str   = "Feedback!";
			t->S.t  += 1;
			t->S.f  += 1.0;
			
			ff_send_out_to(t, 0);  // sends it back
			return GO_ON;
		}
		ff::cout << "Received feedback task #" << ++feedbacktasks <<  std::endl;
		ff_send_out_to(t, 1);  // forward
        return GO_ON;
    }
	long frominputtaks = 0, feedbacktasks = 0;
};

struct Node4: ff_node_t<myTask_t>{
	Node4(long ntasks):ntasks(ntasks) {}
	int svc_init(){
		ff::cout << "Node4 initialized!\n";
		return 0;
	}
    myTask_t* svc(myTask_t* t){
		ff::cout << "Node4: from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";	
		++processed;
		return GO_ON;
    }
	void svc_end() {
		if (processed != ntasks) {
			std::cerr << "ERROR: processed " << processed << " tasks, expected " << ntasks << "\n";
			exit(-1);
		}
		std::cout << "RESULT OK, processed " << processed << " tasks\n";
	}
	long ntasks;
	long processed=0;
};



int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	long ntasks = 20;
	if (argc>1) {
		if (argc != 2) {
			std::cerr << "usage: " << argv[0]
					  << " ntasks\n";
			return -1;
		}
		ntasks = std::stol(argv[1]);
	}
		
    ff_pipeline pipe;
	Node1  n1(ntasks);
	Node2  n2(ntasks);
	Node3  n3;
	Node4  n4(ntasks);
	ff_pipeline pipe1;

	pipe1.add_stage(&n2);
	pipe1.add_stage(&n3);
	pipe1.wrap_around();
	
	pipe.add_stage(&n1);
	pipe.add_stage(&pipe1);
	pipe.add_stage(&n4);
	
    //----- defining the distributed groups ------

    auto G1 = n1.createGroup("G1");
	auto G2 = pipe1.createGroup("G2") << &n2;
	auto G3 = pipe1.createGroup("G3") << &n3;
	auto G4 = n4.createGroup("G4");
	
    // -------------------------------------------	

	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
