/*  
 *                           		  |-> Node31 FdbOutput -->|             
 *             |-> FdbInput Node21 -->|             	      |
 *   Node1---->|             		  |-> Node32 FdbOutput -->| ----> Node4    
 *             |-> FdbInput Node22 -->|                		  |
 *                           		  |-> Node33 FdbOutput -->|
 *
 *             /<--------- a2a0 ---------->/
 *   /<--------------------- pipe --------------------->/
 *
 * G1: pipe0
 * G2: a2a0
 * G3: pipe1
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

struct FdbInput : ff_minode_t<myTask_t>{
	int feedbacks, feedbackTasks = 0;

	FdbInput(int feedbacks) : feedbacks(feedbacks) {}

	myTask_t * svc(myTask_t* t){
		if (fromInput()) return t;
		feedbackTasks++;
		ff::cout << "Received from feeback channel!\n";
		ff_send_out(t);
		return (feedbackTasks == feedbacks ? this->EOS : this->GO_ON);
	}
};


struct Node2: ff_monode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		t->str += std::string(" World");
        return t;
    }
};

struct FdbOutput : ff_monode_t<myTask_t>{
	int feedbacks;
	FdbOutput(int feedbacks) : feedbacks(feedbacks) {}
	myTask_t* svc(myTask_t* t){
		if (t->str == "Hello World"){
			t->str = "Feedback!";
			for(int i = 0; i < feedbacks; i++) ff_send_out_to(new myTask_t(t), i);
			delete t;
			return this->GO_ON;
		}
		ff_send_out_to(t, feedbacks);
		return this->GO_ON;
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
		ff::cout << "Node4: from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";
		++processed;
		delete t;
        return GO_ON;
    }
	void svc_end() {
		if (processed != ntasks) {
			return;
			//abort();
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
	long ntasks = 1;
	if (argc>1) {
		ntasks = std::stol(argv[1]);
	}
		
    ff_pipeline pipe;
	Node1  n1(ntasks);
	Node2  n21, n22;
	Node3  n31, n32, n33;
	Node4  n4(ntasks*2);
	ff_a2a      a2a;
	a2a.add_firstset<ff_node>({new ff_comb(new FdbInput(1), &n21)});
    a2a.add_secondset<ff_node>({new ff_comb(&n31,new FdbOutput(1))});
	a2a.wrap_around();

	pipe.add_stage(&n1);
	pipe.add_stage(&a2a);
	pipe.add_stage(&n4);

	//----- defining the distributed groups ------
	
    auto G1 = n1.createGroup("G1");
	auto G2 = a2a.createGroup("G2");
	auto G3 = n4.createGroup("G3");

    // -------------------------------------------
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
