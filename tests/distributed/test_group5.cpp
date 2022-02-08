/* 
 * FastFlow concurrent network:
 *         
 *    -----------------      ---------------------
 *   | S1 --> | --> W1 |    | --> T1 --> |        |
 *   |        |        |    |            |        |
 *   | S2 --> | --> W2 |--> | --> T2 --> | --> K1 |
 *   |        |        |    |            |        |
 *   | S3 --> | --> W3 |    | --> T3 --> |        |
 *    -----------------      ---------------------
 *   |<----- A2A1 ---->|    |<------ A2A2 ------->|
 *   |<-----------------  pipe ------------------>|
 *
 *  distributed version:
 *
 *          G1                         G3
 *    ----------------        -----------------
 *   |                |      |  T1 -->|        |
 *   | S1 -->| --> W1 |      |        |        |
 *   |       |        | ---> |  T2 -->| --> K1 |
 *   | S2 -->| --> W2 |  --> |        |        |
 *   |                |  |   |  T3 -->|        |
 *    ----------------   |    -----------------
 *           |  ^        |
 *    G2     v  |        |
 *    ----------------   |
 *   |                |  |
 *   | S3 -->| --> W3 |--
 *   |                |
 *    ----------------          
 */

#include <ff/dff.hpp>
#include <mutex>
#include <iostream>

using namespace ff;
std::mutex mtx;

struct S : ff_monode_t<std::string>{

    std::string* svc(std::string* in){
		long outchannels = get_num_outchannels();
		
        for(long i = 0; i < outchannels; i++)
            ff_send_out_to(new std::string("[Task generated from S" + std::to_string(get_my_id()) + " for W" + std::to_string(i) + "]"), i);
        
        return EOS;
    }
};

struct W_left: ff_minode_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << "[W_left" << get_my_id() << " received " << *in << " from S" << get_channel_id() << "]" << std::endl;
		
		return in;
	}
};
struct W_right: ff_monode_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		long outchannels = get_num_outchannels();
		
        for(long i = 0; i < outchannels; i++) 
            ff_send_out_to(new std::string("[Task generated from W_right" + std::to_string(get_my_id()) + " to T" + std::to_string(i) + "]"), i);
		
		return GO_ON;
	}
};

struct T_left: ff_minode_t<std::string> {

    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		std::cout << "[T_left" << get_my_id() << " reiceived " << *in << " from W" << get_channel_id() << "]" << std::endl;
		
		return in;
	}
};
struct T_right: ff_monode_t<std::string> {

    std::string* svc(std::string* in){
		return in;
	}
};

struct K : ff_minode_t<std::string>{
    std::string* svc(std::string* in){
        const std::lock_guard<std::mutex> lock(mtx);
		
        std::cout << "[K received " << *in << " from T" << get_channel_id() << "]" << std::endl;
        delete in;
        return this->GO_ON;
    }
};

int main(int argc, char*argv[]){

	if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}


	using W = ff_comb;
	using T = ff_comb;
	
	ff_pipeline pipe;
    ff_a2a      a2a1;
	ff_a2a      a2a2;
	S   s1, s2, s3;
	W   w1(new W_left, new W_right, true, true);
	W   w2(new W_left, new W_right, true, true);
	W   w3(new W_left, new W_right, true, true);
	T   t1(new T_left, new T_right, true, true);
	T   t2(new T_left, new T_right, true, true);
	T   t3(new T_left, new T_right, true, true);
	K   k;

	pipe.add_stage(&a2a1);
	pipe.add_stage(&a2a2);
	
    a2a1.add_firstset<S>({&s1, &s2, &s3});
    a2a1.add_secondset<W>({&w1, &w2, &w3});

	a2a2.add_firstset<T>({&t1, &t2, &t3});
    a2a2.add_secondset<K>({&k});


	//----- defining the distributed groups ------

	auto g1 = a2a1.createGroup("G1");
	auto g2 = a2a1.createGroup("G2");
	auto g3 = a2a2.createGroup("G3");

	g1 << &s1 << &s2 << &w1 << &w2;
	g2 << &s3 << &w3;
	g3 << &t1 << &t2 << &t3;
	
    // -------------------------------------------

	// running the distributed groups
    if (pipe.run_and_wait_end()<0) {
		error("running a2a\n");
		return -1;
	}
	return 0;
}
