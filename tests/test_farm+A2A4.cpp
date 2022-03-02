/* 
 * FastFlow concurrent network:
 *                   ---------------------------------------------
 *                  |                          pipeA2A1           |
 *                  |     pipe1              -------------------- | 
 *                  | -----------------     |        /<-farm3-->/||
 *                  ||            | W2 |    | T1 ->|        | T4 ||
 *                  || W1-> DefE->| W2 | -->|      |-->T3-->| T4 ||   
 *  ----     -----  ||            | W2 |  ^ | T2 ->|        | T4 ||   ---- 
 * | S1 |-> | S2 |->||    /<-farm1-->/ |  |  -------------------- |  | S3 |
 *  ----    |    |  | ----------------    |     a2a1              |->|    |
 *           ----   |                     |                       |   ----
 *          emitter |                     |                       | collector
 *                  | -----------------   |     a2a2              |      
 *                  ||     /<-farm2-->/|  |  -------------------- |
 *                  ||            | W2 |  v | T1 ->|        | T4 ||
 *                  || W1-> DefE->| W2 | -->|      |-->T3-->| T4 ||
 *                  ||            | W2 |    | T2 ->|        | T4 ||
 *                  | -----------------     |        /<-farm4-->/||
 *                  |     pipe2              -------------------- |
 *                  |                          pipeA2A2           |
 *                   ---------------------------------------------
 *                  /<--------------------a2a ------------------->/
 *          /<------------------------- farm ---------------------------->/
 * /<-------------------------------- pipeMain --------------------------->/ 
 *
 */

#include <ff/ff.hpp>
#include <iostream>
#include <mutex>


using namespace ff;

std::mutex mtx; // for printing purposes

struct S1 : ff_node_t<std::string> {
	S1(long N):N(N) {}
    std::string* svc(std::string*) {
        for(long i = 0; i < N; i++)
            ff_send_out(new std::string("[Task generated from S1 for Worker"));
        
        return EOS;
    }
	long N;
};
	
struct S2 : ff_monode_t<std::string>{
		
	std::string* svc(std::string* in){
		long idx = next % 2;
		std::string* out = new std::string(*in + std::to_string(idx) + "]");
		
		ff_send_out_to(out, idx);
		delete in;
		++next;
        return GO_ON;

    }
	long next=0;
};

struct W1: ff_node_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << "[W1-" << get_my_id() << " received " << *in << " from S2]\n";
		
		return in;
	}
};
struct W2: ff_monode_t<std::string> {

	int svc_init() {
		next = get_my_id();
		return 0;
	}
	
    std::string* svc(std::string* in){
		long outchannels = get_num_outchannels();
		long idx = next % outchannels;

		ff_send_out_to(new std::string("[Task generated from W2-" + std::to_string(get_my_id()) + " to T" + ((idx==0 || idx==1)?std::to_string(idx):std::to_string(idx%3)) + "-" + std::to_string(idx) + "]"), idx);

		++next;
		return GO_ON;
	}
	long next;
};

struct T_left: ff_minode_t<std::string> {	
    std::string* svc(std::string* in){
		return in;
	}
};
struct T_right: ff_monode_t<std::string> {
    std::string* svc(std::string* in){
		return in;
	}
};
struct T_right2: ff_node_t<std::string> {
    std::string* svc(std::string* in){
		return in;
	}
};


struct S3 : ff_minode_t<std::string>{
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		
        std::cout << "[S3 received " << *in << " from T4-" << get_channel_id() << "]" << std::endl;
        delete in;
        return this->GO_ON;
    }
};

int main(int argc, char*argv[]){
	int N=1000;
	if (argc==2) N=std::stol(argv[1]);


	using T1 = ff_comb;
	using T2 = ff_comb;
	using T3 = ff_comb;
	using T4 = T_right2;  // T_right2
	

	ff_pipeline pipe1;
	pipe1.add_stage(new W1, true);
	ff_farm farm1;
	farm1.add_workers({new W2, new W2, new W2});
	farm1.cleanup_workers();
	pipe1.add_stage(&farm1);

	ff_pipeline pipe2;
	pipe2.add_stage(new W1, true);
	ff_farm farm2;
	farm2.add_workers({new W2, new W2, new W2});
	farm2.cleanup_workers();	
	pipe2.add_stage(&farm2);

	ff_pipeline pipeA2A1;
	ff_a2a a2a1;	
    a2a1.add_firstset<ff_comb>( {new T1(new T_left, new T_right, true, true),
								 new T2(new T_left, new T_right, true, true)}, 0, true);
	

	ff_farm farm3;  
	farm3.add_emitter(new T3(new T_left, new T_right, true, true));
	farm3.add_workers({ new T4, new T4, new T4 });

	// to be added in the second set of a2a1, the farm must be inside a pipeline
	ff_pipeline pipefarm1;
	pipefarm1.add_stage(&farm3);

	a2a1.add_secondset<ff_pipeline>({&pipefarm1});
		
	pipeA2A1.add_stage(&a2a1);
	
	ff_pipeline pipeA2A2;
	ff_a2a a2a2;	
	a2a2.add_firstset<ff_comb>( {new T1(new T_left, new T_right, true, true),
								 new T2(new T_left, new T_right, true, true)}, 0, true);	

	ff_farm farm4;
	farm4.add_emitter(new T3(new T_left, new T_right, true, true));
	farm4.add_workers({ new T4, new T4, new T4 });
	// to be added in the second set of a2a2, the farm must be inside a pipeline
	ff_pipeline pipefarm2;
	pipefarm2.add_stage(&farm4);

	a2a2.add_secondset<ff_pipeline>({&pipefarm2});


	pipeA2A2.add_stage(&a2a2);
	
	ff_a2a a2a;
	a2a.add_firstset<ff_pipeline>( {&pipe1,    &pipe2});
    a2a.add_secondset<ff_pipeline>({&pipeA2A1, &pipeA2A2});

	S1 s1(N);
	S2 s2;
	S3 s3;
	
	ff_pipeline pipeMain;
	ff_farm farm;
	farm.add_emitter(&s2);
	farm.add_workers({&a2a});	
	farm.add_collector(&s3);

	pipeMain.add_stage(&s1);	
	pipeMain.add_stage(&farm);
	
    if (pipeMain.run_and_wait_end()<0) {
		error("running a2a\n");
		return -1;
	}
	return 0;
}
