/*
 * FastFlow concurrent network:
 *
 *  -----------------------------------------------------------
 * |  /<------------ a2a(0)----------->/                        |
 * |   -------------------------------                          |
 * |  |                               |                         |
 * |  |  Client ->|                   |                         |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |   |                     |
 * |  |                               |   |                     |
 * |   -------------------------------    |                     |
 * |       ....                           | --> level0Gatherer  |
 * |                                      |                     |
 * |   -------------------------------    |                     |
 * |  |                               |   |                     |
 * |  |  Client ->|                   |   |                     |
 * |  |           | -> level1Gatherer | ->|                     |
 * |  |  Client ->|                   |                         |
 * |  |                               |                         |
 * |   -------------------------------                          |
 * |  /<------------ a2a(n)----------->/                        |
 * |                                                            |
 *  ------------------------------------------------------------
 * /<-------------------------- mainA2A ----------------------->/
 *
 *
 * distributed version:
 *
 *  each a2a(i) is a group, a2a(i) --> Gi i>0
 *  level0Gatherer: G0
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

struct Client: ff_monode_t<myTask_t> {
	Client(long ntasks):ntasks(ntasks) {}
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

struct level1Gatherer: ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		t->str += std::string(" World");
        return t;
    }
};
struct HelperNode: ff_monode_t<myTask_t> {
    myTask_t* svc(myTask_t* t) { return t; }
};

struct level0Gatherer: ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		std::cerr << "level0Gatherer: from (" << get_channel_id() << ") " << t->str << "\n";
		return GO_ON;
    }
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	long ntasks = 100;
	size_t nL  = 3;
	size_t pL  = 2;
	if (argc>1) {
		if (argc != 4) {
			std::cerr << "usage: " << argv[0]
					  << " ntasks n-left-groups pargroup\n";
			return -1;
		}
		ntasks = std::stol(argv[1]);
		nL     = std::stol(argv[2]);
		pL     = std::stol(argv[3]);
	}
		
	ff_a2a mainA2A;
	level0Gatherer root;
	std::vector<ff_node*> globalLeft;

	for(size_t i=0;i<nL; ++i) {
		ff_pipeline* pipe = new ff_pipeline;   // <---- To be removed and automatically added
		ff_a2a* a2a = new ff_a2a;
		pipe->add_stage(a2a, true);
		std::vector<ff_node*> localLeft;
		for(size_t j=0;j<pL;++j)
			localLeft.push_back(new Client(ntasks));
		a2a->add_firstset(localLeft, 0, true);
		a2a->add_secondset<ff_comb>({new ff_comb(new level1Gatherer, new HelperNode, true, true)});

		globalLeft.push_back(pipe);

		// create here Gi groups for each a2a(i) i>0  
		auto g = mainA2A.createGroup("G"+std::to_string(i+1));
		g << pipe;  
		// ----------------------------------------
	}

	mainA2A.add_firstset(globalLeft, true);
	mainA2A.add_secondset<ff_node>({&root});

	// adding the root node as G0
	mainA2A.createGroup("G0") << root;
	// -------------------------------
	
	if (mainA2A.run_and_wait_end()<0) {
		error("running the main All-to-All\n");
		return -1;
	}
	return 0;
}
