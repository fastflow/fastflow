/* 
 * FastFlow concurrent network:
 *                    ---------------------------------------
 *                   |                          pipeA2A1     |
 *                   |                      --------------   | 
 *                   |      pipe1          |  T1 ->|  T3  |  |
 *                   |  -------------      |       |      |  |
 *                   | | W1  --> W2  | --> |       |      |  |       ----
 *  -----------      |  -------------   ^  |  T2 ->|  T4  |  |      |    |
 * | S1 --> S2 | --> |                  |   --------------   | ---->| S3 |
 *  -----------      |  -------------   v   --------------   |       ----
 *     pipe0         | | W1  --> W2  | --> |  T1 ->|  T5  |  |
 *                   |  -------------      |       |      |  |
 *                   |      pipe2          |       |      |  |
 *                   |                     |  T2 ->|  T6  |  |
 *                   |                      --------------   |
 *                   |                          pipeA2A2     |
 *                    ---------------------------------------
 *                                    a2a
 * /<-------------------------- pipeMain ------------------------------->/ 
 *
 *  distributed version:
 *                                     G2
 *                    ---------------------------------------
 *                   |                      --------------   |                        
 *                   |                     |  T1 ->|  T3  |  |
 *                   |  -------------      |       |      |  |        G4
 *     G1            | | W1  --> W2  | --> |       |      |  |       ----
 *  -----------      |  -------------      |  T2 ->|  T4  |  |      |    |
 * | S1 --> S2 | --> |                      --------------   | ---->| S3 |
 *  -----------  |    ---------------------------------------   ^    ----
 *               |                     |  ^                     |
 *               |                     v  |                     |
 *               |    ---------------------------------------   |
 *               |   |  -------------       --------------   |  |     
 *               |   | | W1  --> W2  | --> |  T1 ->|  T5  |  |  |
 *                -->|  -------------      |       |      |  | --
 *                   |                     |       |      |  |
 *                   |                     |  T2 ->|  T6  |  |
 *                   |                      --------------   |
 *                    ---------------------------------------
 *                                       G3
 */

#include <ff/dff.hpp>
#include <mutex>
#include <iostream>

using namespace ff;
std::mutex mtx;

struct S1 : ff_node_t<std::string> {
	S1(long N):N(N) {}
    std::string* svc(std::string*) {
        for(long i = 0; i < N; i++)
            ff_send_out(new std::string("[Task generated from S1 for W"));
        
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

		ff_send_out_to(new std::string("[Task generated from W2-" + std::to_string(get_my_id()) + " to T" + std::to_string(idx) + "]"), idx);

		++next;
		return GO_ON;
	}
	long next;
};

struct T_left: ff_minode_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << "[T" << get_my_id() << " reiceived " << *in << " from W2-" << get_channel_id() << "]" << std::endl;
		
		return in;
	}
};
struct T_right: ff_monode_t<std::string> {

    std::string* svc(std::string* in){
		return in;
	}
};

struct T: ff_minode_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		return in;
	}
};


struct S3 : ff_minode_t<std::string>{
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		
        std::cout << "[S3 received " << *in << " from T" << (3+get_channel_id()) << "]" << std::endl;
        delete in;
        return this->GO_ON;
    }
};

int main(int argc, char*argv[]){

	if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

	int N=10;
	if (argc==2) N=std::stol(argv[1]);


	using T1 = ff_comb;
	using T2 = ff_comb;
	using T3 = T;
	using T4 = T;
	using T5 = T;
	using T6 = T;
	
	ff_pipeline pipe0;
	pipe0.add_stage(new S1(N), true);
	pipe0.add_stage(new S2, true);

	ff_pipeline pipe1;
	pipe1.add_stage(new W1, true);
	pipe1.add_stage(new W2, true);

	ff_pipeline pipe2;
	pipe2.add_stage(new W1, true);
	pipe2.add_stage(new W2, true);

	ff_pipeline pipeA2A1;
	ff_a2a a2a1;	
    a2a1.add_firstset<T1>( {new T1(new T_left, new T_right, true, true),
								new T2(new T_left, new T_right, true, true)}, 0, true);


	a2a1.add_secondset<ff_comb>({new ff_comb(new T3, new T_right, true, true), new ff_comb(new T4, new T_right, true, true)}, true);
	//a2a1.add_secondset<T3>({new T3, new T4}, true);


	pipeA2A1.add_stage(&a2a1);
	
	ff_pipeline pipeA2A2;
	ff_a2a a2a2;	
	a2a2.add_firstset<T1>( {new T1(new T_left, new T_right, true, true),
						new T2(new T_left, new T_right, true, true)}, 0, true);	


	a2a2.add_secondset<ff_comb>({new ff_comb(new T5, new T_right, true, true), new ff_comb(new T6, new T_right, true, true)}, true);
	//a2a2.add_secondset<T3>({new T5, new T6}, true);


	pipeA2A2.add_stage(&a2a2);
	
	ff_a2a a2a;
	a2a.add_firstset<ff_pipeline>( {&pipe1,    &pipe2});
    a2a.add_secondset<ff_pipeline>({&pipeA2A1, &pipeA2A2});

	S3 s3;
	
	ff_pipeline pipeMain;
	pipeMain.add_stage(&pipe0);
	pipeMain.add_stage(&a2a);
	pipeMain.add_stage(&s3);
	//----- defining the distributed groups ------
	auto G1 = pipe0.createGroup("G1");
	auto G2 = a2a.createGroup("G2");
	auto G3 = a2a.createGroup("G3");
	auto G4 = s3.createGroup("G4");

	G2 << &pipe1 << &pipeA2A1;
	G3 << &pipe2 << &pipeA2A2;
	
    // -------------------------------------------

	// running the distributed groups
    if (pipeMain.run_and_wait_end()<0) {
		error("running a2a\n");
		return -1;
	}
	return 0;
}
