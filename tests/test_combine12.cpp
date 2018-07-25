#include <ff/farm.hpp>
#include <ff/pipeline.hpp>
#include <ff/combine.hpp>

using namespace ff;

const long WORKTIME_TICKS=25000;

struct Generator: ff_node_t<long> {
    Generator(long ntasks):ntasks(ntasks) {}
    long *svc(long *) {
	for(long i=1;i<=ntasks;++i)
	    ff_send_out((long*)i);
        return EOS;
    }
    long ntasks;
};

struct Worker1: ff_node_t<long> {
    long* svc(long*in) {
	printf("Worker1 received %ld\n", (long)in);
        return in;
    }
};
struct Worker2: ff_node_t<long> {
    Worker2(const size_t nworkers):nworkers(nworkers) {}
    long* svc(long*in) {
	if (((long)in % nworkers) != (size_t)get_my_id()) {
	    error("WRONG INPUT FOR WORKER%ld, received=%ld\n", get_my_id(), (long)in);
	    abort();
	}
	ticks_wait(WORKTIME_TICKS); 		
        return in;
    }
	const size_t nworkers;
};

struct Emitter: ff_monode_t<long> {
    long *svc(long *in) {
	printf("Emitter received %ld\n", (long)in);
        ff_send_out_to(in, (long)in % get_num_outchannels());
        return GO_ON;
    }
};
struct Collector1: ff_node_t<long> {
    int svc_init() {
	return 0;
    }
    long *svc(long *in) {
	printf("Collector received %ld\n", (long)in);
	if ((long)in != cnt++) abort();

        return in;
    }
    long cnt=1;
};
struct Collector2: ff_node_t<long> {
    long *svc(long *in) {
        return GO_ON;
    }
};


int main(int argc, char* argv[]) {
    int nworkers1=2;
    int nworkers2=3;
    int ntasks=1000;

    if (argc!=1) {
	if (argc!=4) {
	    printf("use: %s nworkers1 nworkers2 ntasks\n", argv[0]);
	    return -1;
	}
	nworkers1 = atoi(argv[1]);
	nworkers2 = atoi(argv[2]);
	ntasks    = atoi(argv[3]);
    }
    
    // first farm, it's an ordered farm
    ff_farm* farm1 = new ff_farm;	
    farm1->add_emitter(new Generator(ntasks));
    std::vector<ff_node*> W1;
    for(int i=0;i<nworkers1;++i)
	W1.push_back(new Worker1);
    farm1->add_workers(W1);
    farm1->add_collector(new Collector1); 
    farm1->cleanup_all();
    farm1->set_ordered();
    
    // second farm
    ff_farm* farm2 = new ff_farm;	
    farm2->add_emitter(new Emitter);    
    farm2->add_collector(new Collector2);
    std::vector<ff_node*> W2;
    for(int i=0;i<nworkers2;++i)
	W2.push_back(new Worker2(nworkers2));
    farm2->add_workers(W2);
    farm2->cleanup_all();

    auto pipe = combine_ofarm_farm(*farm1, *farm2);
    if (pipe.run_and_wait_end()) {
	error("running pipeline\n");
	return -1;
    }
    delete farm1;
    delete farm2;
    
    printf("TEST DONE\n");
    return 0;
}
