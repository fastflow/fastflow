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
    if (argc != 4) {
	std::cerr << "use: " << argv[0] << " name 1|0 host:port\n";
	return -1;
    }
    
    char * name = argv[1];
    char * P = argv[2];      // 1 producer 0 consumer
    char * address = argv[3];  // no check

    // creates the network using 0mq as transport layer
    zmqTransport transport(atoi(P));
    if (transport.initTransport()<0) abort();

    UNICAST Unicast(new UNICAST_DESC(name,&transport,atoi(P)));
    if (!Unicast.init(address)) abort();

    msg_t msg;
    if (atoi(P)) {
	for(int i=0;i<100;++i) {
	    msg.init(new int(i),sizeof(int));
	    Unicast.put(msg);
	}
    } else {
	for(int i=0;i<100;++i) {
	    Unicast.get(msg);
	    printf("received %d\n", *static_cast<int*>(msg.getData()));
	}
    }

    Unicast.close();
    delete Unicast.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
