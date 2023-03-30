/* 
 * FastFlow concurrent network:
 *
 *               --------------------------
 *              |                          |
 *              |              | -> Node2  |
 *              |              |           |
 *  ------      |  -------     | -> Node2  |     -------        -------
 * | DefE | --->| | Node1 | -->|           | -->| Node3 | ---> | Node4 |
 *  ------      |  -------     | -> Node2  |     -------        -------
 * (emitter)    | (emitter)    |           |   (collector)
 *              |              | -> Node2  |
 *              |                          |
 *               --------------------------
 *              /<------- farm ----------->/
 * /<------------------------ farmExt ------------------->/
 * /<---------------------------- pipeMain ---------------------------->/ 
 *
 */


#include <iostream>
#include <ff/ff.hpp>

using namespace ff;


using myTask_t=long;

struct Node1: ff_monode_t<myTask_t>{
	Node1(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t*){
        for(long i=0; i< ntasks; i++) {
            ff_send_out(new myTask_t(i));
		}        
        return EOS;
    }
	const long ntasks;
};

struct Node2: ff_node_t<myTask_t>{
    myTask_t* svc(myTask_t* t){
		std::cout << *t << " (" << get_my_id() << ")\n";
        return t;
    }
};

struct Node3: ff_minode_t<myTask_t> {
	myTask_t* svc(myTask_t* t) {
		std::cout << "Node3 " << *t << " from " << get_channel_id() << "\n";
		return t;
	}
};

struct Node4: ff_node_t<myTask_t>{
	Node4(long ntasks):ntasks(ntasks) {}
    myTask_t* svc(myTask_t* t){
		++processed;
		delete t;
        return GO_ON;
    }
	void svc_end() {
		if (processed != ntasks) {
			std::cout << "ERROR: processed = " << processed << " ntasks= " << ntasks << "\n";
			exit(-1);
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

	
	ff_farm farm;
	farm.add_emitter(new Node1(ntasks));
	farm.add_workers({new Node2, new Node2, new Node2});
	farm.cleanup_emitter();
	farm.cleanup_workers();

	ff_farm farmExt;
	farmExt.add_workers({&farm});
	farmExt.add_collector(new Node3);

	Node4 n4(ntasks);
	
	ff_pipeline pipeMain;
	pipeMain.add_stage(&farmExt);
	pipeMain.add_stage(&n4);

	if (pipeMain.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
