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
	return -1;
    }
    
    char * name = argv[1];
    char * P = argv[2];      // 1 producer 0 consumer
    int    n = atoi(argv[3]);
    char * address = argv[4];  // no check
    
    // creates the network using 0mq as transport layer
    zmqTransport transport(n);
    if (transport.initTransport()<0) abort();
    
    BROADCAST Broadcast(new BROADCAST_DESC(name,(atoi(P)?n:1),&transport,atoi(P)));
    if (!Broadcast.init(address)) abort();

    msg_t msg;
    if (atoi(P)) {
	for(int i=0;i<100;++i) {
	    msg.init(new int(i),sizeof(int));
	    Broadcast.put(msg);
	}
    } else {
	for(int i=0;i<100;++i) {
	    Broadcast.get(msg);
	    printf("received %d\n", *static_cast<int*>(msg.getData()));
	}
    }

    Broadcast.close();
    delete Broadcast.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
