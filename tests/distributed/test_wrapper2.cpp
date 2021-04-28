/*  
 *           | -> Node2 ->|    | -> Node3->Helper ->|
 *   Node1-->|            | -->|                    | --> Node4  
 *           | -> Node2 ->|    | -> Node3->Helper ->|
 *  
 *                              /<-- combine -->/
 *           /<-------------- a2a ---------------->/
 *   /<------------------------ pipe ------------------------>/
 *
 *
 * Node2 uses the WrapperOUT
 * Node3 uses the WrapperIN
 * Helper uses the WrapperOUT
 * Node4 uses the WrapperIN
 */

#include <iostream>
#include <ff/ff.hpp>
#include <ff/distributed/ff_wrappers.hpp>

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
		t->str += std::string(" World");		
		//std::cout << "Node2: " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";

		int dest = ++cnt % get_num_outchannels();
		ff_send_out_to(t, dest);
		
        return GO_ON;
    }
	size_t cnt=0;
};

struct Node3: ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		static bool even=true;
		std::cout << "Node3" << get_my_id()+1 << ": from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";

		t->S.t  += 1;
		t->S.f  += 1.0;
		if (even) {
			ff_send_out(t);
			even=false;
			return GO_ON;
		}
		even=true;
        return t;
    }
};

struct Helper: ff_node_t<myTask_t> {
	myTask_t* svc(myTask_t* t) { return t;}
};

struct Node4: ff_minode_t<myTask_t>{
	Node4(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t* t){
		std::cout << "Node4: from (" << get_channel_id() << ") " << t->str << " (" << t->S.t << ", " << t->S.f << ")\n";
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
	long ntasks = 1000;
	if (argc>1) {
		ntasks = std::stol(argv[1]);
	}

    ff_pipeline pipe;
	Node1 n1(ntasks);
	ff_a2a a2a;
	Node2 n21, n22;
	Node3 n31, n32;
	Node4 n4(ntasks);
	
    pipe.add_stage(&n1);
	std::vector<ff_node*> L;
	std::vector<ff_node*> R;
	L.push_back(new WrapperOUT<true, myTask_t>(&n21, 2));
	L.push_back(new WrapperOUT<true, myTask_t>(&n22, 2));
	a2a.add_firstset(L, 0 ,true);
	
	ff_comb comb1(new WrapperIN<true, myTask_t>(&n31),
				  new WrapperOUT<true,myTask_t>(new Helper,1,true),
				  true,true);	
	ff_comb comb2(new WrapperIN<true, myTask_t>(&n32),
				  new WrapperOUT<true,myTask_t>(new Helper,1,true),
				  true,true);	

	R.push_back(&comb1);
	R.push_back(&comb2);
	a2a.add_secondset(R);
    pipe.add_stage(&a2a);
    pipe.add_stage(new WrapperIN<true, myTask_t>(&n4), true);
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
