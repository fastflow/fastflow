/*
 * FastFlow concurrent network:
 *
 *        ----------------------------------------------------------------
 *       |                                                             |  |
 *       |   | FdbInput-Node2 ->|                 | Node3->FdbOutput ->|  |          
 *        -> |                  |      |   |      |                    |--
 * Node1---->| FdbInput-Node2 ->|->Col1|-->|Col2->| Node3->FdbOutput ->|---> Node4    
 *           |                  |      |   |      |                    |
 *           | FdbInput-Node2 ->|                 | Node3->FdbOutput ->|
 *           
 *           /<------ a2a1------------>/   /<--------- a2a2----------->/
 *       /<---------------------------- pipe1----------------------------->/
 * /<---------------------------------- pipe ----------------------------------->/
 *
 * distributed version:
 *
 * G1: Node1
 * G2: pipe1
 * G3: Node4
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
		t->str += std::string(" World");
        return t;
    }
};
struct Col1: ff_minode_t<myTask_t>{
	myTask_t* svc(myTask_t* in){
		return in;
	}
};
struct Col2: ff_monode_t<myTask_t>{
	myTask_t* svc(myTask_t* in){
		return in;
	}
};

struct Node3: ff_minode_t<myTask_t>{ 
    myTask_t* svc(myTask_t* t){
		t->S.t  += get_my_id();
		t->S.f  += get_my_id()*1.0;
        return t;
    }
};


struct FdbInput : ff_minode_t<myTask_t>{
	FdbInput(long ntasks) : ntasks(ntasks) {}
	
	myTask_t * svc(myTask_t* t){
		if (fromInput()) return t;
		--ntasks;
		//ff::cout << "FdbInput" << get_my_id() << " received from feedback channel: " << t->str << "\n";
		ff_send_out(t);
		return (ntasks>0 ? GO_ON : EOS);
	}
	
	long ntasks;
};

struct FdbOutput : ff_monode_t<myTask_t>{
	int svc_init() {
		feedbacks = get_num_feedbackchannels();
		return 0;
	}

	myTask_t* svc(myTask_t* t){
		if (t->str == "Hello World"){
			t->str = "Feedback!";
			for(int i = 0; i < feedbacks; i++) {
				//std::cout << "FdbOutput" << get_my_id() << " sending task back to " << std::to_string(i) << "\n";
				ff_send_out_to(new myTask_t(t), i);
			}
			delete t;
			return this->GO_ON;
		}
		//std::cout << "FdbOutput" << get_my_id() << " sending task forward into channel " << std::to_string(feedbacks) << "\n";
		ff_send_out_to(t, feedbacks);
		return this->GO_ON;
	}
	int feedbacks=0;
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
	long ntasks = 100;
	size_t nL  = 3;
	size_t nR  = 5;
	if (argc>1) {
		if (argc != 4) {
			std::cerr << "usage: " << argv[0]
					  << " ntasks num-left num-right\n";
			return -1;
		}
		ntasks = std::stol(argv[1]);
		nL     = std::stol(argv[2]);
		nR     = std::stol(argv[3]);
	}
		
    ff_pipeline pipe;
	Node1  n1(ntasks);
	ff_pipeline pipe1;
	ff_a2a a2a1;
	ff_a2a a2a2;
	Node4  n4(ntasks*nL);

	pipe1.add_stage(&a2a1);
	pipe1.add_stage(&a2a2);
	pipe1.wrap_around();
	
	pipe.add_stage(&n1);
	pipe.add_stage(&pipe1);
	pipe.add_stage(&n4);

	std::vector<ff_node*> L;
	for(size_t i=0;i<nL;++i)
		L.push_back(new ff_comb(new FdbInput(ntasks), new Node2, true, true));
	a2a1.add_firstset(L, 0, true);
	a2a1.add_secondset<Col1>({new Col1}, true);
	
	std::vector<ff_node*> R;
	for(size_t i=0;i<nR;++i)
		R.push_back(new ff_comb(new Node3, new FdbOutput, true, true));
	a2a2.add_firstset<Col2>({new Col2}, 0, true);
    a2a2.add_secondset(R, true);
	
	
    //----- defining the distributed groups ------

    auto G1 = n1.createGroup("G1");
	auto G2 = pipe1.createGroup("G2");
	auto G3 = n4.createGroup("G3");
	
    // -------------------------------------------	

	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
