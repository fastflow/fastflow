#include <cstdio>
#include <string>
#include <ff/ff.hpp>

/* 
 * Simple test implementing the multi-reader single-writer protocol 
 * using a farm building block.
 *
 * The pattern is called All-or-Node in 
 * "State Access Patterns in Stream Parallel Computations”. IJHPCA, 32:6, doi: 10.1177/1094342017694134
 *
 * Logical schema of the test:
 *                         -------------
 *                        |             |
 *                        |      |--> Worker -->|
 *                        v      |              |
 *     Generator --> Scheduler-->|              |--> Gatherer
 *                        ^      |              |
 *                        |      |--> Worker -->|
 *                        |             |
 *                         -------------
 *
 *                    |<----------- All-or-None ------------> |
 *
 */
/*
 * Author: Massimo Torquati
 */
using namespace ff;

const int NUM_WRITES_PER_ITERATION = 7;

// this is the state accessed using the multiple-reader single-writer protocol.
static long some_global_state=0;  // it will store the number of writes


// my stream item data type
struct Task_t {
    Task_t(bool RW):RW(RW) {}
    bool RW; // true Read false Write

    /* other stuff here.... */
};

// generator of the stream
struct Generator: ff_node_t<Task_t> {
    Generator(long niter):niter(niter) {}
    Task_t* svc(Task_t*) {
	// NOTE: if you change the number of writes please remember to
	// update NUM_WRITES_PER_ITARATION 
	for(int i=0;i<niter;++i) {  
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(false));   // write  (1)
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(false));   // write  (2)
	    ff_send_out(new Task_t(false));   // write  (3)
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(true));    // read
	    ff_send_out(new Task_t(false));   // write  (4)
	    ff_send_out(new Task_t(false));   // write  (5)
	    ff_send_out(new Task_t(true));    // read  
	    ff_send_out(new Task_t(false));   // write  (6)
	    ff_send_out(new Task_t(false));   // write  (7)
	}
	return EOS; // generating the End-Of-Stream
    }
    long niter;
};

// last stage of the pipeline
struct Gatherer: ff_minode_t<Task_t> {
    Task_t* svc(Task_t*) {	return GO_ON;   }
    void svc_end() {
	printf("number of writes %ld\n", some_global_state);
    }
};

// Generic worker. It reads and writes the global state. 
struct Worker: ff_monode_t<Task_t> {
    Task_t* svc(Task_t* item) {

	// ---- Worker's business logic code -----------------
	//
	// this is just a dummy example, here you have to do something smart.
	if (!item->RW) {
	    long tmp = some_global_state;
	    some_global_state=0;
	    usleep(5000);
	    some_global_state = tmp + 1;
	} else 
	    usleep(500);   // .... reads take more time
	//
	// ---------------------------------------------------
	
	ff_send_out_to(item, 1);  // sending the result forward
	ff_send_out_to(item, 0);  // sending an 'ack message' back to the Scheduler
	return GO_ON;
    }
};

// this is the node implementing the read/write logic
struct Scheduler: ff_node_t<Task_t> {
    Scheduler(ff_loadbalancer* lb):lb(lb) {}

    Task_t* svc(Task_t* item) {
	if (lb->get_channel_id() <0) { // item coming from the previous stage
	    if (item == nullptr) {
		struct timespec ts{0,0};
		nanosleep(&ts,nullptr); // some small backoff time here!
		return GO_ON;
	    }
	    if (item->RW) { // it's a read
		ff_send_out(item); // sending the item to one Worker
		outstanding_reads++;
	    } else { // it's a write
		ff_node::input_active(false); // stop receiving from the previous stage
		buffered_write = item;        // remember the operation that has to be done
		
		if (outstanding_reads==0) {   // the write can be sent to one of the Workers
		    ff_send_out(buffered_write);
		    outstanding_writes=1;
		}
	    }
	    return GO_ON; // keep me alive
	}
	// item coming from one of the Workers
	if (item->RW) { // ... it's a read ack
	    outstanding_reads--;
	    if (outstanding_reads==0) {
		if (buffered_write) { // we have a waiting Write
		    
		    ff_send_out(buffered_write); // sending the Write
		    buffered_write=nullptr;
		    outstanding_writes=1;

		    return GO_ON; // the input from the previous stage is still disabled
		} 
	    }
	    // checking termination
	    if (have_to_terminate()) return EOS;
	    return GO_ON;
	}
	// ... it's a write ack
	assert(outstanding_writes==1);
	outstanding_writes=0;
	ff_node::input_active(true); // restart receiving from the previous stage
	buffered_write=nullptr;

	if (have_to_terminate()) return EOS;
	return GO_ON;
    }

    bool have_to_terminate() {
	return (eos_arrived && outstanding_reads==0 && outstanding_writes==0);
    }

    void eosnotify(ssize_t id) {
	if (id<0) { // the EOS is coming from the previous stage
	    eos_arrived=true;
	    if (have_to_terminate()) lb->broadcast_task(EOS);
	}
    }
    
    ff_loadbalancer* lb;
    long outstanding_reads=0;
    long outstanding_writes=0;
    Task_t* buffered_write=nullptr;

    bool eos_arrived = false;
};

int main(int argc, char* argv[]) {
    long niter = 10;
    if (argc > 1) {
	niter=std::stol(std::string(argv[1]));
    }
    
    Generator Gen(niter);
    Gatherer  Gat;

    const size_t nworkers = 1; // 3

    ff_farm farm;
    std::vector<ff_node*> W;    
    for(size_t i=0;i<nworkers;++i)
	W.push_back(new Worker);
    farm.add_workers(W);
    Scheduler sched(farm.getlb());
    farm.add_emitter(&sched);
    farm.remove_collector();
    farm.wrap_around();
    farm.cleanup_workers();
    ff_Pipe<> pipe(Gen, farm, Gat);

    pipe.setXNodeInputQueueLength(512,true);  
    pipe.setXNodeOutputQueueLength(512,true); 
    if (pipe.run_and_wait_end()<0) {
	error("running pipe\n");
	return -1;
    }

    // checking the result
    if (some_global_state != niter*NUM_WRITES_PER_ITERATION) {
	error("WRONG RESULT\n");
	return -1;
    } else
	printf("RESULT OK\n");
    return 0;
}




