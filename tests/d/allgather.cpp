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
    int    n = atoi(argv[3]);  // NOTE: It is the num peers for the consumer, and the node id for the producers
    char * address = argv[4];  // no check
    
    // creates the network using 0mq as transport layer
    zmqTransport transport(n);
    if (transport.initTransport()<0) abort();
    
    ALLGATHER Gather(new ALLGATHER_DESC(name,(atoi(P)?1:n),&transport,atoi(P)));
    if (!Gather.init(address)) abort();

    if (atoi(P)) {
	ALLGATHER::tosend_t msg;
	for(int i=0;i<100;++i) {
	    msg.init(new int(i),sizeof(int));
	    Gather.put(msg);
	}
    } else {
	ALLGATHER::torecv_t msg;	
	for(int i=0;i<100;++i) {
	    Gather.get(msg);
	    assert(msg.size()==(size_t)n);
	    for(int j=0;j<n;++j)
		printf("received %d\n", *static_cast<int*>(msg[j].getData()));
	}
    }
    
    Gather.close();
    delete Gather.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
