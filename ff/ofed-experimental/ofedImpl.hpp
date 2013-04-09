/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file rdmaImpl.hpp
 *  \brief Communication patterns in distributed FastFlow using ØMQ
 */

#ifndef _FF_OFEDIMPL_HPP_
#define _FF_OFEDIMPL_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */

#include <cstdlib>
#include <cstdio>
#include <sstream>
#include <string>
#include <vector>
#include <assert.h>

#include <ff/svector.hpp>
#include <ff/d/ofedTransport.hpp>

namespace ff {


/* ---------------- Pattern macro definitions -------------------- */

#define UNICAST                          commPattern<ofed1_1>
#define UNICAST_DESC(name,trasp,P)       UNICAST::descriptor(name,1,trasp,P)
#define BROADCAST                        commPattern<ofedBcast>
#define BROADCAST_DESC(name,n,trasp,P)   BROADCAST::descriptor(name,n,trasp,P)
#define ALLGATHER                        commPattern<ofedAllGather>
#define ALLGATHER_DESC(name,n,trasp,P)   ALLGATHER::descriptor(name,n,trasp,P)
#define FROMANY                          commPattern<ofedFromAny>
#define FROMANY_DESC(name,n,trasp,P)     FROMANY::descriptor(name,n,trasp,P)
#define SCATTER                          commPattern<ofedScatter>
#define SCATTER_DESC(name,n,trasp,P)     SCATTER::descriptor(name,n,trasp,P)
#define ONETOMANY                        commPattern<ofed1_N>
#define ONETOMANY_DESC(name,n,trasp,P)   ONETOMANY::descriptor(name,n,trasp,P)
#define ONDEMAND                         commPattern<ofedOnDemand>
#define ONDEMAND_DESC(name,n,trasp,P)    ONDEMAND::descriptor(name,n,trasp,P)

/* TODO */
#define MANYTOONE                        commPattern<ofedN_1>
#define MANYTOONE_DESC(name,n,trasp,P)   MANYTOONE::descriptor(name,n,trasp,P)

/* --------------------------------------------------------------- */

// used in Unicast, Bcast, Scatter, On-Demand, OneToMany


pthread_mutex_t descriptorsMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t descriptorsCond = PTHREAD_COND_INITIALIZER;
int descriptors = 0;
int descCounter = 0;
int closedDescriptors = 0;

class descriptor {
private:
	ofedContext* 			 s_ctx;
	int 					 descId;
	const std::string        name;          // name of descriptor (?)
	ofedConnection**	     id;
	ff::ofedTransport* const transport;
	const bool               passive;
	const int                peers;
	bool					 sender;

public:
	/**
	 * Constructor. Creates a One-to-Many descriptor, that is, instantiate a
	 * transport layer using the firt node as a router
	 *
	 * @param[in] name name of the descriptor
	 * @param[in] peers number of peers
	 * @param[in] transport pointer to the transport layer object
	 * @param[in] P=true if passive node
	 * @param[in] sender=true if data sender node
	 *
	 */
	descriptor(const std::string name, const int peers, ff::ofedTransport* const transport, const bool passive, const bool sender) :
				name(name), id(NULL), transport(transport), passive(passive), peers(peers), sender(sender)
	{
		ofedTDBG(printf("entering CONSTRUCTOR descriptor name %s, peers %d, passive %d\n", name.c_str(), this->peers, passive));
		id = (ofedConnection**) malloc(sizeof(ofedConnection*)*this->peers);
		// s_ctx = (ofedContext*) malloc(sizeof(ofedContext));
		// memset(s_ctx, 0, sizeof(ofedContext));

		// give to each desctiptor a single numeric ID
		pthread_mutex_lock(&descriptorsMutex);
		descId = descriptors;
		descriptors ++;
		pthread_mutex_unlock(&descriptorsMutex);

		s_ctx = transport->getContext(descId);


		// for each peer, add an endpoint. newEndPoint must be thread safe
		for (int i=0; i<this->peers; i++)
			id[i] = transport->newEndPoint(passive, sender, s_ctx);

		ofedTDBG(printf("exiting CONSTRUCTOR descriptor\n"));
	}

	~descriptor() {
		ofedTDBG(printf("entering DESTRUCTOR descriptor\n"));
		// free(id);
	 	// free(s_ctx);
		ofedTDBG(printf("exiting DESTRUCTOR descriptor\n"));
	}

	ofedConnection** getId() {
		return id;
	}

	ofedContext* getCtx() {
		return s_ctx;
	}

	int getDescId() {
		return descId;
	}

	int close() {
		ofedTDBG(printf("entering descriptor::close()\n"));

		for (int i=0; i<peers; i++) {
			ofedTDBG(printf("descriptor1_N::close peer %d closing\n", i));
			id[i]->prepareDisconnection();
		}

		ofedTDBG(printf("exiting descriptor::close()\n"));
		int ret = 0;

		return ret;
	}

	/*
	 *  Initialise a socket
	 *
	 *  addr is the ip address in the form ip/host:port
	 *  nodeId is the node identifier in the range [0..inf[
	 */
	inline int init(const std::string& addr, const int nodeId) {
		ofedTDBG(printf("entering descriptor::init nodeId: %d\n", nodeId));
		// give to each desctiptor a single numeric ID

		//  Find the ':' that separates hostname name from service.
		size_t found = addr.find(':');

		if (!found)
			return -1;

		std::string hostname = addr.substr(0,found);
		std::string service = addr.substr(found+1);
		ofedTDBG(printf("descriptor: hostname: %s, port: %s\n", hostname.c_str(), service.c_str()));

		if (passive) {
			ofedTDBG(printf("descriptor: passive side (server)\n"));

			struct sockaddr_in addr_in;

			memset(&addr_in, 0, sizeof(addr_in));
			addr_in.sin_family = AF_INET;

			ofedTDBG(printf("descriptor: calling wait for event\n"));

			for (int p=0; p<peers; p++) {
				uint16_t port = atoi(service.c_str());
				ofedTDBG(printf("descriptor: listening on port %d.\n", port+p));

				addr_in.sin_port = htons(port+p);
				TEST_NZ(rdma_bind_addr(id[p]->id, (struct sockaddr *)&addr_in));

				TEST_NZ(rdma_listen(id[p]->id, 10)); /* backlog=10 is arbitrary */
				port = ntohs(rdma_get_src_port(id[p]->id));
				ofedTDBG(printf("descriptor: call create thread.\n"));
			}

			for (int p=0; p<peers; p++) {
				transport->waitForEvent(id[p]->id, RDMA_CM_EVENT_ESTABLISHED, s_ctx);
			}

			ofedTDBG(printf("descriptor: after waitForEvent\n"));

		} else {
			ofedTDBG(printf("descriptor: active side (client)\n"));
			struct addrinfo *addrinfo = NULL;
			TEST_NZ(getaddrinfo(hostname.c_str(), service.c_str(), NULL, &addrinfo));

			ofedTDBG(printf("descriptor: calling wait for event\n"));

			for (int i=0; i<10; i++) {
				TEST_NZ(rdma_resolve_addr(id[0]->id, NULL, addrinfo->ai_addr, TIMEOUT_IN_MS));

				ofedTDBG(printf("descriptor: try now\n"));
				if (transport->waitForEvent(id[0]->id, RDMA_CM_EVENT_ESTABLISHED, s_ctx))
					break;
				sleep (2);

				rdma_disconnect(id[0]->id);
				transport->deleteEndPoint(id[0]);
				transport->resetContext(descId);
				delete id[0];
				id[0] = transport->newEndPoint(passive, sender, s_ctx);

				if (i==9) {
					ofedTDBG(printf("descriptor: too many retries\n"));
					die("Couldn't connect to Server");
				}
			}

			freeaddrinfo(addrinfo);
		}

		waitAll();

		ofedTDBG(printf("descriptor: calling initConnection\n"));
		for (int p=0; p<peers; p++)
			id[p]->initConnection();

		waitAll();

		ofedTDBG(printf("descriptor: calling rdmaRecv\n"));
		for (int p=0; p<peers; p++)
			id[p]->rdmaRecv();
		waitAll();

		ofedTDBG(printf("descriptor: calling startCqPoller\n"));
		ofedTransport::readCqThreadParams p;
		p.s_ctx = s_ctx;
		p.transport = transport;
		p.id = descId;
		p.connections = id;
		p.peers = peers;

		transport->startCqPoller(&p, descId);
		waitAll();

		ofedTDBG(printf("descriptor: calling registerMemory\n"));
		for (int p=0; p<peers; p++)
			id[p]->registerMemory();

		waitAll();

		ofedTDBG(printf("descriptor: calling startSendThread\n"));
		if (descId == 0)
			transport->startSendThread();

		waitAll();

		ofedTDBG(printf("exiting descriptor::init\n"));
		return 0;
	}

	void waitAll() {
		pthread_mutex_lock(&descriptorsMutex);
		descCounter ++;
		if (descId != 0)
			pthread_cond_wait(&descriptorsCond, &descriptorsMutex);

		pthread_mutex_unlock(&descriptorsMutex);


		if (descId == 0) {
			while (descCounter < descriptors)
				usleep(1000);
			descCounter = 0;
			pthread_cond_signal(&descriptorsCond);
		}
	}
};

/**
 *  \struct descriptor1_N
 *
 *  \brief Descriptor used in several communication patterns:
 *  Unicast, Broadcast, Scatter, On-Demand, OneToMany.
 *
 *  REW - Complete this one
 */
class descriptor1_N {
private:
	const std::string        	name;          // name of descriptor (?)
	ofedConnection**	     	id;
	ff::ofedTransport* const 	transport;
public:
	const bool               	P;
private:
	const int               	peers;
	bool						rdmaHdrSet;
	descriptor* 				desc;
	ofedContext* 				s_ctx;
	int 						descId;


public:

	/**
	 * Constructor. Creates a One-to-Many descriptor, that is, instantiate a
	 * transport layer using the firt node as a router
	 *
	 * @param[in] name name of the descriptor
	 * @param[in] peers number of peers
	 * @param[in] transport pointer to the transport layer object
	 * @param[in] P=true if sender
	 *
	 */
	descriptor1_N(const std::string name, const int peers, ff::ofedTransport* const transport, const bool P ) :
				name(name), id(NULL), transport(transport), P(P), peers(P ? peers : 1), rdmaHdrSet(false), desc(NULL)
	{
		ofedTDBG(printf("entering CONSTRUCTOR descriptor1_N name %s, peers %d, passive %d\n", name.c_str(), this->peers, P));
		desc = new descriptor(this->name, this->peers, this->transport, this->P, this->P);

		id = desc->getId();
		s_ctx = desc->getCtx();
		descId = desc->getDescId();

		ofedTDBG(printf("exiting CONSTRUCTOR descriptor1_N\n"));
	}

	~descriptor1_N() {
		ofedTDBG(printf("entering DESTRUCTOR descriptor1_N\n"));
		delete desc;
		ofedTDBG(printf("exiting DESTRUCTOR descriptor1_N\n"));
	}

	inline int close() {
		ofedTDBG(printf("entering descriptor1_N::close\n"));
		ofedTDBG(printf("exiting descriptor1_N::close\n"));
		return desc->close();
	}

	/*
	 *  Initialise a socket
	 *
	 *  addr is the ip address in the form ip/host:port
	 *  nodeId is the node identifier in the range [0..inf[
	 */
	inline int init(const std::string& addr, const int nodeId=-1) {
		ofedTDBG(printf("entering descriptor1_N::init nodeId: %d\n", nodeId));
		if (!id) return -1;
		ofedTDBG(printf("exiting descriptor1_N::init nodeId: %d\n", nodeId));
		return desc->init(addr, nodeId);
	}

	// broadcast
	inline bool send(const msg_t& msg, int flags = 0) {
		ofedTDBG(printf("entering descriptor1_N::send unicast\n"));
		ofedTDBG(printf("descriptor1_N::send SENDING CONTENT\n"));

		bool ret = true;
		for (int i=0; i<peers; i++)
			id[i]->send(msg);

        ofedTDBG(printf("exiting descriptor1_N::send unicast\n"));
		return ret;
	}

	// broadcast
	inline bool sendmore(const msg_t& msg, int flags) {
		ofedTDBG(printf("entering descriptor1_N::sendmore unicast\n"));
		bool ret = true;
		for (int i=0; i<peers; i++)
			id[i]->send(msg);

        ofedTDBG(printf("exiting descriptor1_N::sendmore unicast\n"));
		return ret;
	}

	// unicast
	inline bool send(const msg_t& msg, const int dest, int flags) {
		ofedTDBG(printf("entering descriptor1_N::send unicast\n"));

        assert(dest<peers);
		ofedTDBG(printf("descriptor1_N::send SENDING CONTENT\n"));

		bool ret = true;
		id[dest]->send(msg);

        ofedTDBG(printf("exiting descriptor1_N::send unicast\n"));
		return ret;
	}

	// unicast
	inline bool sendmore(const msg_t& msg, const int dest, int flags) {
		ofedTDBG(printf("entering descriptor1_N::sendmore unicast\n"));
		bool ret = true;

		id[dest]->send(msg);
        ofedTDBG(printf("exiting descriptor1_N::sendmore unicast\n"));
		return ret;
	}

	// receives a message from a RDMA_DEALER
	inline bool recvhdr(msg_t& msg) {
		ofedTDBG(printf("entering descriptor1_N::recvhdr\n"));
		bool ret = true;
		id[0]->recv(msg);
		ofedTDBG(printf("exiting descriptor1_N::recvhdr\n"));
		return ret;
	}

	// receives a message after having received the header
	inline bool recv(msg_t& msg) {
		ofedTDBG(printf("entering descriptor1_N::recv\n"));
		bool ret = true;
		id[0]->recv(msg);
		ofedTDBG(printf("exiting descriptor1_N::recv\n"));
		return ret;
	}

	// returns the total number of peers (NOTE: if !P,  peers is 1)
	inline const int getPeers() const { return peers;}

    // In the on-demand pattern it is used by the producer (i.e. the on-demand consumer)
    inline bool sendReq() {
        return sendreq();
    }

    inline bool sendreq() {
    	id[0]->sendReq();
        return true;
    }

    // In the on-demand pattern it is used by the consumer (i.e. the on-demand producer)
    inline bool recvReq(int& peer, int flags=0) {
        msg_t from;
        msg_t msg;

		while(peer == -1) {
			for(int i = 0; i < peers; ++i) {
				if (id[i]->tryRecvReq()) {
					peer = i;
					break;
				}
			}
			usleep(1000);
		}
		id[peer]->recvReq();

        return true;
    }
}; // end descriptor1_N


/**
 *  \struct descriptorN_1
 *
 *  \brief Descriptor used in several communication patterns:
 *  AllGather, Collect FromAny, ManyToOne.
 *
 */
class descriptorN_1 {
private:
	const std::string        	name;          // name of descriptor (?)
	ofedConnection**	     	id;
	ff::ofedTransport* const 	transport;
public:
	const bool               	P;
private:
	const int               	peers;
	descriptor* 				desc;
	ofedContext* 				s_ctx;
	int 						descId;
	int							toRecv;
	bool						recvHdr;

public:
	/// Constructor
	descriptorN_1( const std::string name, const int peers, ff::ofedTransport* const transport, const bool P ) :
				name(name), id(NULL), transport(transport), P(P), peers(P?1:peers)
	{
		toRecv = 0;
		recvHdr = true;
		ofedTDBG(printf("entering CONSTRUCTOR descriptorN_1 name %s, peers %d, passive %d\n", name.c_str(), this->peers, P));

		desc = new descriptor(this->name, this->peers, this->transport, !this->P, this->P);
		id = desc->getId();
		s_ctx = desc->getCtx();
		descId = desc->getDescId();

		ofedTDBG(printf("exiting CONSTRUCTOR descriptorN_1\n"));
	}

	~descriptorN_1() {
		ofedTDBG(printf("entering DESTRUCTOR descriptor1_N\n"));
		delete desc;
		ofedTDBG(printf("exiting DESTRUCTOR descriptor1_N\n"));
	}

	inline int close() {
		ofedTDBG(printf("entering descriptorN_1::close\n"));
		ofedTDBG(printf("exiting descriptorN_1::close\n"));
		return desc->close();
	}


	// @param addr is the ip address in the form ip/host:port
	// nodeId is the node identifier in the range [0..inf[
	//
	inline int init(const std::string& addr, const int nodeId=-1) {
		ofedTDBG(printf("entering descriptorN_1::init nodeId: %d\n", nodeId));
		if (!id) return -1;

		ofedTDBG(printf("exiting descriptorN_1::init\n"));
		return desc->init(addr, nodeId);
	}

	inline bool send(const msg_t& msg, int flags = 0) {
		ofedTDBG(printf("entering descriptorN_1::send\n"));
		ofedTDBG(printf("descriptorN_1::send SENDING CONTENT\n"));

		bool ret = true;
		ofedTDBG(printf("descriptorN_1::send sending size: %d\n", (int) ((msg_t&) msg).size()));
		id[0]->send(msg);
		ofedTDBG(printf("descriptorN_1::send sent size: %d\n", (int) ((msg_t&) msg).size()));

        ofedTDBG(printf("exiting descriptorN_1::send\n"));
		return ret;
	}

	// receives a message from a RDMA_ROUTER
	/*
	inline bool recvhdr(msg_t& msg, int& peer) {
		ofedTDBG(printf("entering descriptorN_1::recvhdr\n"));

		bool ret = true;

		ofedTDBG(printf("descriptorN_1::recvhdr peer %d\n", toRecv));

		id[toRecv]->recv(msg);

		peer = toRecv;
		toRecv = (toRecv + 1) % peers;

		ofedTDBG(printf("exiting descriptorN_1::recvhdr\n"));
		return ret;
	}
	*/
	inline bool recvhdr(msg_t& msg, int& peer) {
		ofedTDBG(printf("entering descriptorN_1::recvhdr\n"));

		bool ret = true;

		ofedTDBG(printf("descriptorN_1::recvhdr peer %d\n", toRecv));

		peer = 0;
		while (true) {
			if (id[peer]->tryRecv())
				break;

			peer = (peer + 1) % peers;
		}
		id[peer]->recv(msg);
		ofedTDBG(printf("exiting descriptorN_1::recvhdr\n"));
		return ret;
	}

	// receives a message after having received the header
	inline bool recv(msg_t& msg, int& peer) {
		ofedTDBG(printf("entering descriptorN_1::recv\n"));

		bool ret = true;

		ofedTDBG(printf("descriptorN_1::recv peer %d\n", toRecv));

		if (peer == -1) {
			peer = 0;
			while (true) {
				if (id[peer]->tryRecv())
					break;

				peer = (peer + 1) % peers;
			}
		}
		id[peer]->recv(msg);
		ofedTDBG(printf("exiting descriptorN_1::recv\n"));
		return ret;
	}

	inline bool recvreq() {
        msg_t useless;

        id[0]->recv(useless);

		return true;
	}

	inline bool done(bool sendready=false) {
		ofedTDBG(printf("entering descriptorN_1::done\n"));
        if (sendready) {
            static const char ready[] = "READY";
            for(int i = 0; i < peers; ++i) {
                msg_t msg(ready, 6);
                // msg_t to(const_cast<char*>(transportIDs[i].c_str()), transportIDs[i].length()+1, 0, 0);

                // id[i]->send(to);
                id[i]->send(msg);
            }
        }
        ofedTDBG(printf("exiting descriptorN_1::done\n"));
        recvHdr = true;
        return true;
	}

	// returns the number of partners of the single communication (NOTE: if P, peers is 1)
	inline const int getPeers() const { return peers;}
}; // end descriptorN_1

/*!
 *  \class ofed1_1
 *
 *  \brief RDMA implementation of the 1 to 1 communication patter
 */
class ofed1_1 {
public:
	typedef ofedTransport      TransportImpl;   /// to the transpoort layer
	typedef msg_t              tosend_t;        /// ut supra
	typedef msg_t              torecv_t;        /// ut supra
	typedef descriptor1_N      descriptor;      ///< descriptor used in Unicast, Bcast, Scatter,
	///< On-Demand, OneToMany

	enum {MULTIPUT=0 };

	ofed1_1():desc(NULL),active(false) {}

	// point to point communication implemented using ØMQ
	ofed1_1(descriptor* D):desc(D),active(false) {}

	/// Set the descriptor
	inline void setDescriptor(descriptor* D) {
		if (desc)  desc->close();
		desc = D;
	}

	/// Get the descriptor
	inline  descriptor* getDescriptor() { return desc; }

	/** Initialize the communication pattern */
	inline bool init(const std::string& address,const int nodeId=-1) {
		ofedTDBG(printf("entering ofed1_1::init\n");)
		if (active) return false;
		// we force 0 to be the nodeId for the consumer
		ofedTDBG(printf("ofed1_1::init call desc->init \n");)
		if(!desc->init(address,(desc->P)?nodeId:0))
			active = true;

		ofedTDBG(printf("exiting ofed1_1::init\n");)
		return active;
	}

	// sends one message
	inline bool put(const tosend_t& msg) { return desc->send(msg, 0, 0); }
#define RDMA_NOBLOCK 1
#define RDMA_SNDMORE 2
	inline bool putmore(const tosend_t& msg) { return desc->sendmore(msg,0,RDMA_SNDMORE);}

	inline bool put(const msg_t& msg, const int) { return desc->send(msg, 0, 0); }
	inline bool putmore(const msg_t& msg, const int) { return desc->sendmore(msg, 0, RDMA_SNDMORE); }

	// receives the message header (should be called before get)
	inline bool gethdr(torecv_t& msg, int& peer) { peer=0; return desc->recvhdr(msg); }

	// receives one message
	inline bool get(torecv_t& msg, int=0) { return desc->recv(msg); }

	// returns the number of distinct messages for one single communication
	inline const int getToWait() const { return 1;}
	inline const int putToPerform() const { return 1;}

	inline void done() {}

	// close pattern
	inline bool close() {
		if (!active) return false;
		if (!desc->close()) return false;
		active=false;
		return true;
	}

protected:
	descriptor* desc;
	bool        active;
};



/*!
 *  \class ofedBcast
 *
 *  \brief RDMA implementation of the broadcast communication patter
 */
class ofedBcast {
public:
    // typedef rdmaTransportMsg_t msg_t;
    typedef ff::ofedTransport  TransportImpl;
    typedef msg_t              tosend_t;
    typedef msg_t              torecv_t;
    typedef descriptor1_N      descriptor;

    enum {MULTIPUT=0};

    ofedBcast():desc(NULL),active(false) {}
    ofedBcast(descriptor* D):desc(D),active(false) {}

    // sets the descriptor
    inline void setDescriptor(descriptor* D) {
        if (desc)  desc->close();
        desc = D;
    }

    // returns the descriptor
    inline  descriptor* getDescriptor() { return desc; }

    // initializes communication pattern
    inline bool init(const std::string& address,const int nodeId=-1) {
        if (active) return false;
        if(!desc->init(address,nodeId)) active = true;
        return active;
    }

    // sends one message
    inline bool put(const tosend_t& msg) { return desc->send(msg, 0); }

    inline bool putmore(const tosend_t& msg) { return desc->sendmore(msg,RDMA_SNDMORE);}


    inline bool put(const msg_t& msg, const int to) {
        return desc->send(msg, to, 0);
    }

    inline bool putmore(const msg_t& msg, const int to) {
        return desc->sendmore(msg,to,RDMA_SNDMORE);
    }

    // receives the message header (should be called before get)
    inline bool gethdr(torecv_t& msg, int& peer) { peer=0; return desc->recvhdr(msg); }

    // receives one message
    inline bool get(torecv_t& msg, int=0) { return desc->recv(msg); }

    // returns the number of distinct messages for one single communication
    inline const int getToWait() const { return 1;}
    inline const int putToPerform() const { return 1;}

    // close pattern
    inline bool close() {
        if (!active) return false;
        if (!desc->close()) return false;
        active=false;
        return true;
    }

    inline void done() {}

protected:
    descriptor* desc;
    bool        active;
};
//
//
////
////  RDMA implementation of the ALL_GATHER communication patter
////
//
/*!
 *  \class ofedAllGather
 *
 *  \brief RDMA implementation of the ALL_GATHER communication patter
 */
class ofedAllGather {
public:
    typedef ff::ofedTransport  TransportImpl;
    typedef msg_t              tosend_t;
    typedef svector<msg_t>     torecv_t;
    typedef descriptorN_1      descriptor;

    enum {MULTIPUT=0 };

    ofedAllGather():desc(NULL),active(false) {}
    ofedAllGather(descriptor* D):desc(D),active(false) {}

    // sets the descriptor
    inline void setDescriptor(descriptor* D) {
        if (desc)  desc->close();
        desc = D;
    }

    // returns the descriptor
    inline  descriptor* getDescriptor() { return desc; }

    // initializes communication pattern
    inline bool init(const std::string& address,const int nodeId=-1) {
        if (active) return false;
        if(!desc->init(address,nodeId)) active = true;
        if (!desc->P) done();
        return active;
    }

    // sends one message
    inline bool put(const tosend_t& msg) {
    	ofedTDBG(printf("entering ofedAllGather::put\n"));
    	bool ret1 = desc->recvreq();
        if (!ret1) return false;

        ofedTDBG(printf("ofedAllGather::put sending size: %d\n", (int) ((msg_t&) msg).size()));
        bool ret2 = desc->send(msg, 0);
        ofedTDBG(printf("ofedAllGather::put sending size: %d\n", (int) ((msg_t&) msg).size()));

        ofedTDBG(printf("exiting ofedAllGather::put\n"));
        return ret2;
    }

    inline bool putmore(const tosend_t& msg) {
    	ofedTDBG(printf("entering ofedAllGather::putmore\n"));
    	ofedTDBG(printf("ofedAllGather::putmore sending size: %d\n", (int) ((msg_t&) msg).size()));
    	bool ret = desc->send(msg,RDMA_SNDMORE);
    	ofedTDBG(printf("ofedAllGather::putmore sending size: %d\n", (int) ((msg_t&) msg).size()));
    	ofedTDBG(printf("exiting ofedAllGather::putmore\n"));
    	return ret;
    }

    inline bool put(const msg_t& msg, const int) {
        return put(msg);
    }

    inline bool putmore(const tosend_t& msg, const int) {
        return putmore(msg);
    }

    // receives the message header ONLY from one peer (should be called before get)
    inline bool gethdr(msg_t& msg, int& peer) {
        return desc->recvhdr(msg, peer);
    }

    // receives one message ONLY from one peer (gethdr should be called before)
    inline bool get(msg_t& msg, int& peer) {
        return desc->recv(msg,peer);
    }
    // receives one message
    inline bool get(torecv_t& msg) {
        const int peers=desc->getPeers();
        int useless = 0;
        msg.resize(peers);
        for(int i=0;i<peers;++i)
            if (!desc->recv(msg[i],useless)) return false;
        done();
        return true;
    }

    // returns the number of distinct messages for one single communication
    inline const int getToWait() const { return desc->getPeers();}
    inline const int putToPerform() const { return 1;}

    inline void done() { desc->done(true); }

    // close pattern
    inline bool close() {
        if (!active) return false;
        if (desc->P) if (!desc->recvreq()) return false;
        if (!desc->close()) return false;
        active=false;
        return true;
    }

protected:
    descriptor* desc;
    bool        active;
};


/*!
 *  \class ofedFromAny
 *
 *  \brief RDMA implementation of the collect from ANY communication patter
 */
class ofedFromAny {
public:
//     typedef rdmaTransportMsg_t  msg_t;
    typedef ff::ofedTransport  TransportImpl;
    typedef msg_t              tosend_t;
    typedef msg_t              torecv_t;
    typedef descriptorN_1      descriptor;

    enum {MULTIPUT=0 };

    ofedFromAny():desc(NULL),active(false) {}
    ofedFromAny(descriptor* D):desc(D),active(false) {}

    // sets the descriptor
    inline void setDescriptor(descriptor* D) {
        if (desc)  desc->close();
        desc = D;
    }

    // returns the descriptor
    inline  descriptor* getDescriptor() { return desc; }

    // initializes communication pattern
    inline bool init(const std::string& address,const int nodeId=-1) {
        if (active) return false;
        if(!desc->init(address,nodeId)) active = true;
        return active;
    }

    // sends one message
    inline bool put(const tosend_t& msg) {
        return desc->send(msg, 0);

    }

    inline bool putmore(const tosend_t& msg) {
        return desc->send(msg,RDMA_SNDMORE);
    }

    inline bool put(const msg_t& msg, const int) {
        return desc->send(msg, 0);
    }

    inline bool putmore(const tosend_t& msg, const int) {
        return desc->send(msg,RDMA_SNDMORE);
    }

    // receives the message header (should be called before get)
    inline bool gethdr(msg_t& msg, int& peer) {
        return desc->recvhdr(msg,peer);
    }

    // receives one message
    inline bool get(msg_t& msg, int& peer) {
        return desc->recv(msg,peer);
    }
    // receives one message
    inline bool get(msg_t& msg) {
        int useless=-1;
        return desc->recv(msg,useless);
    }

    // returns the number of distinct messages for one single communication
    inline const int getToWait() const { return 1;}
    inline const int putToPerform() const { return 1;}

    inline void done() { desc->done(); }

    // close pattern
    inline bool close() {
        if (!active) return false;
        if (!desc->close()) return false;
        active=false;
        return true;
    }

protected:
    descriptor* desc;
    bool        active;
};


/*!
 *  \class ofedScatter
 *
 *  \brief RDMA implementation of the SCATTER communication patter
 */
class ofedScatter {
public:
    typedef ff::ofedTransport       TransportImpl;
    typedef svector<msg_t>     tosend_t;
    typedef msg_t              torecv_t;
    typedef descriptor1_N      descriptor;

    enum {MULTIPUT=1 };

    ofedScatter():desc(NULL),active(false) {}
    ofedScatter(descriptor* D):desc(D),active(false) {}

    // sets the descriptor
    inline void setDescriptor(descriptor* D) {
        if (desc)  desc->close();
        desc = D;
    }

    // returns the descriptor
    inline  descriptor* getDescriptor() { return desc; }

    // initializes communication pattern
    inline bool init(const std::string& address,const int nodeId=-1) {
        if (active) return false;
        if(!desc->init(address,nodeId)) active = true;
        return active;
    }

    // sends one message
    inline bool put(const tosend_t& msg) {
        const int peers=desc->getPeers();
        assert(msg.size()==(size_t)peers);
        for(int i=0;i<peers;++i) {
            if (!desc->send(msg[i], i, 0)) return false;
        }
        return true;
    }
    // TODO
    inline bool putmore(const tosend_t& msg) {
        return -1;
    }

    inline bool put(const msg_t& msg, const int to) {
        return desc->send(msg, to, 0);
    }
    inline bool put(const msg_t& msg) {
        return -1;
    }

    inline bool putmore(const msg_t& msg, const int to) {
        return desc->sendmore(msg,to,RDMA_SNDMORE);
    }
    inline bool putmore(const msg_t& msg) {
        return -1;
    }

    // receives the message header (should be called before get)
    inline bool gethdr(torecv_t& msg, int& peer) { peer=0; return desc->recvhdr(msg); }

    // receives one message
    inline bool get(torecv_t& msg, int=0) { return desc->recv(msg); }

    // returns the number of distinct messages for one single communication
    inline const int getToWait() const { return 1;}
    inline const int putToPerform() const { return desc->getPeers();}

    // close pattern
    inline bool close() {
        if (!active) return false;
        if (!desc->close()) return false;
        active=false;
        return true;
    }

    inline void done() {}

protected:
    descriptor* desc;
    bool        active;
};


/*!
 *  \class ofedOnDemand
 *
 *  \brief RDMA implementation of the ON-DEMAND communication patter
 */
class ofedOnDemand {
protected:
    inline bool sendReq() {
        if (!desc->sendReq()) return false;
        requestSent=true;
        return true;
    }

public:
    typedef ff::ofedTransport  TransportImpl;
    typedef msg_t              tosend_t;
    typedef msg_t              torecv_t;
    typedef descriptor1_N      descriptor;

    enum {MULTIPUT=0 };

    ofedOnDemand():desc(NULL),active(false),to(-1),requestsToWait(0),requestSent(false) {}
    ofedOnDemand(descriptor* D):desc(D),active(false),to(-1),requestsToWait(0),requestSent(false) {}

    // sets the descriptor
    inline void setDescriptor(descriptor* D) {
        if (desc)  desc->close();
        desc = D;
    }

    // returns the descriptor
    inline  descriptor* getDescriptor() { return desc; }

    // initializes communication pattern
    inline bool init(const std::string& address,const int nodeId=-1) {
        if (active) return false;
        if(!desc->init(address,nodeId)) {
            if (!desc->P) if (!sendReq()) return false;
            active = true;
        }

        To.resize(desc->getPeers());
        for(int i=0;i<desc->getPeers();++i) To[i]=false;

        return active;
    }

    // sends one message
    inline bool put(const tosend_t& msg) {
        // FIX: It would be better to use a non-blocking calls to receive requests, and
        //      in case recvReq returns false the msg should be inserted into a local queue.
        //      It is required an extra threads for handling incoming requests.....
        if (to<0) if (!desc->recvReq(to)) return false;
        int r = desc->send(msg, to, 0);
        to=-1;
        return r;
    }

    // sends one message, more messages have to come
    inline bool putmore(const tosend_t& msg) {
        // see FIX comment above
        if (to<0) if (!desc->recvReq(to)) return false;
        return desc->sendmore(msg, to, RDMA_SNDMORE);
    }

    inline bool put(const tosend_t& msg, const int dest, int flag=0) {

        if (dest<0) {  // sends the same message to all peers, usefull for termination
            for(int i=0;i<desc->getPeers();++i) {
                to = i;
                tosend_t _msg;
                _msg.copy(const_cast<msg_t&>(msg));
                if (!desc->send(_msg, to, flag)) { to=-1; return false;}
                ++requestsToWait;
            }
            return true;
        }
        if (To[dest]) {
            if (!desc->send(msg, dest, flag)) { to=-1; return false;}
            To[dest]=false;
            return true;
        }
        do {
        	to = dest; // TODO: Without this, other sides stay connected.
            if (!desc->recvReq(to)) { to=-1; return false;}

            if (to == dest) {
                if (!desc->send(msg, to, flag)) { to=-1; return false;}
                return true;
            }
            assert(To[to]==false);
            To[to]=true;
        } while(1);

        // not reached
        return false;
    }
    inline bool putmore(const tosend_t& msg, const int dest) {
        return put(msg,dest,RDMA_SNDMORE);
    }

    // receives the message header (should be called before get)
    inline bool gethdr(torecv_t& msg, int& peer) {
        peer=0;
        bool r = desc->recvhdr(msg);
        if (r) sendReq();
        return r;
    }

    // receives one message part
    inline bool get(torecv_t& msg) {
        bool r=desc->recv(msg);
        if (!requestSent && r) sendReq();
        return r;
    }
    // receives one message part
    inline bool get(torecv_t& msg,int& peer) {
        peer=0;
        return get(msg);
    }

    // close pattern
    inline bool close() {
        if (!active) return false;
        if (desc->P) {
            int useless = 0;
            requestsToWait+=desc->getPeers();
            for(int i=0;i<desc->getPeers();++i)
                if (To[i]) --requestsToWait;
            for(int i=0;i<requestsToWait;++i) {
                if (!desc->recvReq(useless)) return false;
            }
        }
        if (!desc->close()) return false;
        active=false;
        return true;
    }

    // returns the number of distinct messages for one single communication
    inline const int getToWait() const { return 1;}
    inline void done() {requestSent=false; }

protected:
    descriptor*   desc;
    bool          active;
    int           to;
    int           requestsToWait;
    bool          requestSent;
    svector<bool> To;
};
//
//
//
///*
// *  \class ofed1_N
// *
// *  \brief RDMA implementation of the ONE_TO_MANY communication patter.
// *
// *  This pattern can be used to dynamically change among UNICAST, BROADCAST
// *  and SCATTER patterns.
// *
// *  ***TO BE COMPLETED AND TESTED***
// */
//class ofed1_N {
//public:
//    typedef rdmaTransportMsg_t msg_t;
//    typedef svector<msg_t>     tosend_t;
//    typedef msg_t              torecv_t;
//    typedef descriptor1_N      descriptor;
//
//    enum {MULTIPUT=0 };
//
//    rdma1_N():desc(NULL),active(false) {}
//    rdma1_N(descriptor* D):desc(D),active(false) {}
//
//    // sets the descriptor
//    inline void setDescriptor(descriptor* D) {
//        if (desc)  desc->close();
//        desc = D;
//    }
//
//    // returns the descriptor
//    inline  descriptor* getDescriptor() { return desc; }
//
//    // initializes communication pattern
//    inline bool init(const std::string& address,const int nodeId=-1) {
//        if (active) return false;
//        if(!desc->init(address,nodeId)) active = true;
//        return active;
//    }
//
//    // sends one message
//    inline bool put(const tosend_t& msg) {
//        const int peers=desc->getPeers();
//        assert(msg.size()==(size_t)peers);
//        for(int i=0;i<peers;++i)
//            if (!desc->send(msg[i], i, 0)) return false;
//        return true;
//    }
//
//    // sends one message.
//    // if @param to is equal to -1 it sends the @param msg
//    // in broadcast to all connected peers
//    inline bool put(const msg_t& msg, const int to) {
//        return desc->send(msg, to, to);
//    }
//
//    inline bool putmore(const msg_t& msg, const int to) {
//        return desc->sendmore(msg,to,RDMA_SNDMORE);
//    }
//
//    // receives the message header (should be called before get)
//    inline bool gethdr(torecv_t& msg, int& peer) { peer=0; return desc->recvhdr(msg); }
//
//    // receives one message
//    inline bool get(torecv_t& msg, int=0) { return desc->recv(msg); }
//
//    // close pattern
//    inline bool close() {
//        if (!active) return false;
//        if (!desc->close()) return false;
//        active=false;
//        return true;
//    }
//
//    inline void done() {}
//
//protected:
//    descriptor* desc;
//    bool        active;
//};
//
///*!
// *
// * @}
// */
//
//
}

#endif // _FF_OFEDIMPL_HPP_
