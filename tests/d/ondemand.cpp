#include <assert.h>
#include <cstdlib>
#include <iostream>
#include <string>
#include <ff/utils.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

const unsigned MAX_SLEEP=500000;

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
    char * P = argv[2];      // 1 producer 0 consumer
    int    n = atoi(argv[3]);
    char * address = argv[4];  // no check
    
    // creates the network using 0mq as transport layer
    zmqTransport transport(n);
    if (transport.initTransport()<0) abort();
    
    ONDEMAND OnDemand(new ONDEMAND_DESC(name,n,&transport,atoi(P)));
    if (!OnDemand.init(address)) abort();

    srandom(::getpid()+(getusec()%4999));

    msg_t msg;
    if (atoi(P)) {
	for(int i=0;i<10;++i) {
	    msg.init(new int(i),sizeof(int));
	    OnDemand.put(msg);
	}
	// sending EOS
	//msg.init(new int(-1),sizeof(int));
	//OnDemand.put(msg,-1);
	for(int i=0;i<n;++i) {
	    msg_t eos;
	    eos.init(new int(-1),sizeof(int));
	    OnDemand.put(eos,i);
	}
    } else {
	int ntasks=0, useless;
	do {
	    // here we call gethdr instead of calling get() and done()
	    if (!OnDemand.gethdr(msg,useless)) break;

	    int * d = static_cast<int*>(msg.getData());
	    if (*d==-1) { printf("RECEIVED EOS\n"); break; }
	    long sleeptime= (random() % MAX_SLEEP) + n*1000000;
	    printf("received %d sleeping for %ld us\n", *static_cast<int*>(msg.getData()), sleeptime);
	    usleep(sleeptime);
	    ++ntasks;
	} while(1);
	printf("got %d tasks\n", ntasks);
    }

    OnDemand.close();
    delete OnDemand.getDescriptor();
    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
