#include <cstdio>
#include <iostream>

#include <ff/ff.hpp>

using namespace ff;

struct first: ff_node_t<long> {
    long *svc(long *) {
	for(long i=0;i<10;++i) 
	    ff_send_out(new long(i));
	
	return EOS;
    }
} First;

struct last: ff_node_t<long> {
    long *svc(long *in) {
	printf("received %ld\n", *in);
	return GO_ON;
    }
} Last;


int main() {
    
    ff_Pipe<> pipe(First, Last);
    
    pipe.run_and_wait_end();
    pipe.ffStats(std::cout);
    
    return 0;
}
