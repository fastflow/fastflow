/*
 *
 *             host0                      host1
 *       ----------------             ----------------
 *      |                |           |                |
 *      |                |  UNICAST  |                |
 *    --|---> InOut0  ---|---------- |---> InOut1 --- |--
 *   |  |                |           |                |  |
 *   |  |                |           |                |  |
 *   |   ----------------             ----------------   |
 *   |                     UNICAST                       |
 *    ---------------------------------------------------
 *
 *
 */

#include <iostream>
#include <ff/node.hpp>
#include <ff/dnode.hpp>
#include <ff/d/inter.hpp>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>

using namespace ff;

#define COMM1  zmq1_1
#define COMM2  zmq1_1

#define NUMTASKS 1000

class InOut0: public ff_dinout<COMM1,COMM2> {
protected:
    static void callback(void * e,void* ) {
	delete ((long*)e);
    }
public:
    typedef COMM1::TransportImpl        transport_t;

    InOut0(const std::string& name1, const std::string& address1, 
	  const std::string& name2, const std::string& address2,
	  transport_t* const transp):
	name1(name1),address1(address1),
	name2(name2),address2(address2),
	transp(transp) {
    }

    // initializes dnode
    int svc_init() {
	ff_dinout<COMM1,COMM2>::initOut(name2, address2, 1, transp,0,callback);  
	ff_dinout<COMM1,COMM2>::initIn(name1, address1, 1, transp);
	return 0;
    }

    void * svc(void *task) {	
	if (task==NULL)  {
	    for(long i=1;i<=NUMTASKS;++i)
		ff_send_out((void*)new int(i));
	    ff_send_out((void*)FF_EOS);
	    return GO_ON;
	}
	printf("InOut0 received %ld\n", *(long*)task);
	return GO_ON;
    }

    void svc_end() {
	printf("InOut0 ending\n");
    }

protected:
    const std::string name1;
    const std::string address1;
    const std::string name2;
    const std::string address2;
    transport_t   * transp;
};


class InOut1: public ff_dinout<COMM1,COMM2> {
protected:
    static void callback(void * e,void* ) {
	delete ((long*)e);
    }
public:
    typedef COMM1::TransportImpl        transport_t;

    InOut1(const std::string& name1, const std::string& address1, 
	  const std::string& name2, const std::string& address2,
	  transport_t* const transp):
	name1(name1),address1(address1),
	name2(name2),address2(address2),
	transp(transp) {
    }

    // initializes dnode
    int svc_init() {
	ff_dinout<COMM1,COMM2>::initIn(name2, address2, 1, transp);  
	ff_dinout<COMM1,COMM2>::initOut(name1, address1, 1, transp, 0, callback);  
	return 0;
    }

    void * svc(void *task) {	
	printf("InOut1 received %ld\n", *(long*)task);
	return (new long(*(long*)task)); 
    }

    void svc_end() {
	printf("InOut1 ending\n");
    }

    virtual FFBUFFER * get_out_buffer() const { return (FFBUFFER*)1;}

protected:
    const std::string name1;
    const std::string address1;
    const std::string name2;
    const std::string address2;
    transport_t   * transp;
};


int main(int argc, char * argv[]) {    
    if (argc != 6) {
	std::cerr << "use: " << argv[0] << " name1 name2 1|0 host0:port host1:port\n";
	return -1;
    }
    
    char * name1    = argv[1];
    char * name2    = argv[2];
    char * P        = argv[3];  // 1 producer 0 consumer
    char * address1 = argv[4];  // no check
    char * address2 = argv[5];  // no check

    // creates the network using 0mq as transport layer
    zmqTransport transport(0);
    if (transport.initTransport()<0) abort();
    
    if (atoi(P)) {
	InOut0 n0(name1,address1,name2,address2,&transport);
	n0.skipfirstpop(true);
	n0.run();
	n0.wait();
    } else {
	InOut1 n1(name1,address1,name2,address2,&transport);
	n1.run();
	n1.wait();
    }

    transport.closeTransport();
    std::cout << "done\n";
    return 0;
}
