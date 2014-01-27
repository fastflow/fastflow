#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

typedef zmqTransportMsg_t msg_t;

int main(int argc, char * argv[]) {    
    if (argc != 5) {
	std::cerr << "use: " << argv[0] << " name 1|0 nhosts master-host:port\n";
	std::cerr << "  1|0   : 1 for the master 0 for other hosts\n";
	std::cerr << "  nhosts: is the number of hosts for the master and the hostID for the others\n";
	return -1;
    }
    
    char * name = argv[1];
    char * P = argv[2];        // 1 producer 0 consumer
    int    n = atoi(argv[3]);  // num peers for the producer, node id for the consumers
    char * address = argv[4];  // no check
    
    // creates the network using 0mq as transport layer
    zmqTransport transport(n);
    if (transport.initTransport()<0) abort();
    
    SCATTER Scatter(new SCATTER_DESC(name,n,&transport,atoi(P)));
    if (!Scatter.init(address)) abort();

    msg_t msg;
    if (atoi(P)) {	
	for(int i=0;i<100;++i) {
	    SCATTER::tosend_t msg;
	    msg.resize(n);
	    int *M = new int[n];
	    for(int j=0;j<n;++j) {
		M[j]=i+j;
		msg[j].init(&M[j],sizeof(int));
	    }
	    Scatter.put(msg);
	}
    } else {
	for(int i=0;i<100;++i) {
	    Scatter.get(msg);
	    printf("received %d\n", *static_cast<int*>(msg.getData()));
	}
    }

    Scatter.close();
    delete Scatter.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
