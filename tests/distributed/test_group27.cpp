/*
 * FastFlow concurrent network:
 *
 *  /<------------- pipe -------------->/
 *            /<-------- a2a -------->/
 *              
 *                      |            | 
 *           |-> Node2->| -> Node3 ->|
 *    ------>|          |            | -- 
 *    |      |-> Node2->| -> Node3 ->|   |
 *    |                 |            |   |
 *    |                                  |
 *     ----------------------------------
 *
 *
 * distributed version:
 *
 *  Node2 - Node3  [upper]=> G1
 *  Node2 - Node3  [lower]=> G3
 */

#define TORECEIVE 2

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

struct Node2: ff_monode_t<myTask_t>{
	Node2(int index) : index(index) {}
    int svc_init(){
        ff::cout << "[Node2] initilized!\n";
        return 0;
    }
    myTask_t* svc(myTask_t* t){
        if (t == nullptr){
            // generate a task 
            //ff::cout << "[Node2] generating!\n";
            ff_send_out_to(new myTask_t, index);
            return GO_ON;
        }
        //ff::cout << "[Node2] received_something!\n";
        delete t;
        if (++received == (TORECEIVE - 1)) {
            ff::cout << "[Node2] Sending EOS!!\n";
            return EOS;
        }
        return GO_ON;
	}

    void eosnotify(ssize_t id){
        ff::cout << "[Node2] Received EOS from " << id << std::endl;
    }

    void svc_end(){
        ff::cout << "[Node2] Ended (svc_end)\n";
    }

    int index;
    int received = 0;
};


struct Node3: ff_monode_t<myTask_t>{ 
    Node3(int index): index(index){}
    int svc_init(){
        //ff::cout << "[Node3] initilized!\n";
        return 0;
    }
    myTask_t* svc(myTask_t* t){
		// distributore
        //ff::cout << "[Node3] distributing!\n";
        for(int i = 0; i < TORECEIVE; i++)
            if (i != index) ff_send_out_to(new myTask_t(t),i);
        delete t;
        return GO_ON;
    }
    
    void eosnotify(ssize_t id){
        ff::cout << "[Node3] Received EOS from " << id << std::endl;
    }

    void svc_end(){
        ff::cout << "[Node3] Ended (svc_end)\n";
    }

    int index;
};

struct AdapterSx : ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){ 
        ff::cout << "[AdapterSx] Received something (addr: " << t << ")\n";
        return t;
    }

    void eosnotify(ssize_t id){
        //ff::cout << "[AdapterSx] Received EOS from " << id << std::endl;
    }

    void svc_end(){
        ff::cout << "[AdapterSx] Ended (svc_end)\n";
    }
};

struct AdapterDx : ff_minode_t<myTask_t>{
    myTask_t* svc(myTask_t* t){ 
        ff::cout << "[AdapterDx] Received something (addr: " << t << ")\n";
        return t;
    }

    void eosnotify(ssize_t id){
        ff::cout << "[AdapterDx] Received EOS from " << id << std::endl;
    }

    void svc_end(){
        ff::cout << "[AdapterDx] Ended (svc_end)\n";
    }
};


int main(int argc, char*argv[]){
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	long ntasks = 100;
	if (argc>1) {
		if (argc != 2) {
			std::cerr << "usage: " << argv[0]
					  << " ntasks\n";
			return -1;
		}
		ntasks = std::stol(argv[1]);
	}

	ff_pipeline pipe;
	ff_a2a a2a;
    a2a.wrap_around();

    auto n2_1 = ff_comb(new AdapterSx, new Node2(0));
    auto n2_2 = ff_comb(new AdapterSx, new Node2(1));

    auto n3_1 = ff_comb(new AdapterDx, new Node3(0));
    auto n3_2 = ff_comb(new AdapterDx, new Node3(1));

	a2a.add_firstset<ff_node>({&n2_1, &n2_2}, 0);
	a2a.add_secondset<ff_node>({&n3_1, &n3_2}); 

    a2a.createGroup("G1") << &n2_1 << &n3_1;
    a2a.createGroup("G2") << &n2_2 << &n3_2;
    //pipe.add_stage(&a2a);
    //pipe.wrap_around();

	if (a2a.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}

	return 0;
}
