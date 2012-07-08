#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

int main(int argc, char * argv[]) {    
    if (argc != 5) {
	std::cerr << "use: " << argv[0] << " name 1|0 nhosts master-host:port\n";
	return -1;
    }
    
    char * name = argv[1];
    char * P = argv[2];        // 1 producer 0 consumer
    int    n = atoi(argv[3]);  // num peers for the consumer node id for the producers
    char * address = argv[4];  // no check
    
    // creates the network using 0mq as transport layer
    zmqTransport transport(n);  // NOTE: n is used as node id
    if (transport.initTransport()<0) abort();
    
    FROMANY FromAny(new FROMANY_DESC(name,(atoi(P)?1:n),&transport,atoi(P)));
    if (!FromAny.init(address)) abort();

    if (atoi(P)) {
	FROMANY::tosend_t msg;
	for(int i=0;i<100;++i) {
	    msg.init(new int(i),sizeof(int));
	    FromAny.put(msg);
	}
    } else {
	FROMANY::torecv_t msg;	
	for(int i=0;i<100*n;++i) {
	    FromAny.get(msg);
	    printf("received %d\n", *static_cast<int*>(msg.getData()));
	}
    }
    
    FromAny.close();
    delete FromAny.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
