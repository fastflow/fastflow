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

/*!
 *  \file readTransport.hpp
 *  \brief Definition of the external transport layer based on infiniband
 */

#ifndef _FF_OFEDTRANSPORT_HPP_
#define _FF_OFEDTRANSPORT_HPP_

#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <deque>
#include <algorithm>
#include <netdb.h>
#include <stdlib.h>
#include <fcntl.h>
#include <poll.h>
#include <unistd.h>
#include <string>
#include <cstdio>
#include <queue>
#include <cerrno>
#include <sys/time.h>
#include <ff/d/ofedTools.hpp>
#include <ff/d/ofedConnection.hpp>
#include <rdma/rdma_cma.h>

const int RDMA_BUFFER_SIZE = BUFFER_SIZE;

pthread_mutex_t buildContextMutex = PTHREAD_MUTEX_INITIALIZER;
pthread_cond_t buildContextCond = PTHREAD_COND_INITIALIZER;

namespace ff {

/*!
 * \class ofedTransport
 */
class ofedTransport {

private:
	enum {NUM_IO_THREADS=1};

	pthread_t sendThread;
	pthread_t cqPollerThread[20];
	bool cqPollerThreadEnded[20];
	unsigned int wcPolled[20];

	pthread_mutex_t initMutex;
	pthread_mutex_t readCqThreadMutex;
	pthread_cond_t readCqThreadCond;
	bool sendThreadStarted;
	bool sendThreadEnded;
	int threadCounter;
	int exitedThreadCounter;
	ofedContext* contexts[20];


protected:
	const int                	procId;    	 // Process (or thread) ID
	std::deque<ofedConnection*> connections; // all active end-points

	// closes all connections
	inline int closeConnections() {

		for (int i=0; i<threadCounter; i++) {
			while (!cqPollerThreadEnded[i])
			 	usleep(1000);
		}

		waitSendThread();
		for (unsigned int i=0; i<connections.size(); i++)
			rdma_disconnect(connections[i]->id);

		for (unsigned int i=0; i<connections.size(); i++)
			waitForEvent(connections[i]->id, RDMA_CM_EVENT_DISCONNECTED, connections[i]->getContext());

		for(unsigned i=0;i<connections.size();++i) {
			connections[i]->destroyConnection();

			delete connections[i];
			connections[i]=NULL;
		}

		for (int i=0; i<threadCounter; i++)
			free(contexts[i]);

		return 0;
	}

public:
	typedef ff::msg_t msg_t;

	ofedTransport(const int procId) :
		procId(procId) {

		pthread_mutex_init(&initMutex, NULL);
		pthread_mutex_init(&readCqThreadMutex, NULL);
		pthread_cond_init(&readCqThreadCond, NULL);

		for (int i=0; i<20; i++)
			cqPollerThreadEnded[i] = false;

		for (int i=0; i<20; i++)
			wcPolled[i] = 0;

		sendThreadStarted = false;
		sendThreadEnded = false;
		threadCounter = 0;
		exitedThreadCounter = 0;
	};

	~ofedTransport() {
		// closeTransport();
	}

	ofedContext* getContext(int id) {
		ofedContext* s_ctx = (ofedContext*) malloc(sizeof(ofedContext));
		memset(s_ctx, 0, sizeof(ofedContext));
		contexts[id] = s_ctx;

		return s_ctx;
	}

	void resetContext(int id) {
		contexts[id]->ctx = NULL;
		TEST_NZ(ibv_dealloc_pd(contexts[id]->pd));
		TEST_NZ(ibv_destroy_cq(contexts[id]->cq));
		TEST_NZ(ibv_destroy_comp_channel(contexts[id]->comp_channel));
	}



	/**
	 * Initialise the transport layer: create a new ØMQ context
	 * and clean the list of active en-points.
	 *
	 * \returns 0 if successful
	 */
	int initTransport() {
		ofedTDBG(printf("entering initTransport\n"));

		ofedTDBG(printf("exiting initTransport\n"));
		return 0;
	}

	/**
	 * Close the transport layer, close all connections to any
	 * active endpoint and delete the existing context.
	 *
	 * \returns 0 if successful
	 */
	int closeTransport() {
		closeConnections();
		return 0;
	}

	struct readCqThreadParams {
		ofedTransport* transport;
		ofedContext* s_ctx;
		ofedConnection** connections;
		int peers;
		int id;
	};

	void startCqPoller(readCqThreadParams* params, int id) {
		ofedTDBG(printf("entering startCqPoller\n"));
		TEST_NZ(pthread_create(&cqPollerThread[id], NULL, readCqEvents, (void*) params));
		ofedTDBG(printf("startCqPoller: CQ POLLER THREAD STARTED \n"));
		ofedTDBG(printf("exiting startCqPoller\n"));
	}

	void startSendThread() {
		ofedTDBG(printf("entering startSendThread\n"));
		pthread_mutex_lock(&initMutex);
		if (!sendThreadStarted) {
			TEST_NZ(pthread_create(&sendThread, NULL, sendFromThread, (void*) this));

			ofedTDBG(printf("startSendThread: SEND THREAD STARTED \n"));
		}
		sendThreadStarted = true;
		pthread_mutex_unlock(&initMutex);
		ofedTDBG(printf("exiting startSendThread\n"));
	}

	/**
	 * Create a new endpoint and push it to the active list.
	 *
	 */
	ofedConnection* newEndPoint(const bool passive, const bool sender, ofedContext* s_ctx) {
		ofedTDBG(printf("entering newEndPoint\n"));

		rdma_event_channel *ec = NULL;
		rdma_cm_id* id = NULL;

		TEST_Z(ec = rdma_create_event_channel());
		TEST_NZ(rdma_create_id(ec, &id, NULL, RDMA_PS_TCP));

		if (!id) {
			return NULL;
		}

		ofedConnection* c = new ofedConnection(s_ctx, !passive, sender);

		c->id = id;
		pthread_mutex_lock(&initMutex);
		connections.push_back(c);
		pthread_mutex_unlock(&initMutex);
		ofedTDBG(printf("exiting newEndPoint\n"));

		return c;
	}

	/**
	 * Delete the id pointed by \p s. It removes the id from
	 * the list of active ids and then destroys it.
	 */
	int deleteEndPoint(ofedConnection* s) {
		ofedTDBG(printf("entering deleteEndPoint\n"));

		if (s) {
			ofedTDBG(printf("deleting endpoing\n"));
			std::deque<ofedConnection*>::iterator it = std::find(connections.begin(), connections.end(), s);
			pthread_mutex_lock(&initMutex);
			if (it != connections.end()) {

				ofedTDBG(printf("destroying connection\n"));
				connections.erase(it);

				pthread_mutex_unlock(&initMutex);
				s->destroyConnection();

				return 0;
			}
			pthread_mutex_unlock(&initMutex);
		}
		ofedTDBG(printf("exiting deleteEndPoint\n"));

		return -1;
	}

	inline void buildContext(ibv_context* verbs, ofedContext* s_ctx)
	{
		ofedTDBG(printf("entering buildContext\n"));
		pthread_mutex_lock(&buildContextMutex);
		if (s_ctx->ctx) {
			pthread_mutex_unlock(&buildContextMutex);
			return;
		}

		s_ctx->ctx = verbs;
		s_ctx->ctx->num_comp_vectors = NUM_COMP_VECTORS;

		TEST_Z(s_ctx->pd = ibv_alloc_pd(s_ctx->ctx));
		TEST_Z(s_ctx->comp_channel = ibv_create_comp_channel(s_ctx->ctx));
		TEST_Z(s_ctx->cq = ibv_create_cq(s_ctx->ctx, CQE_ENTRIES, NULL, s_ctx->comp_channel, 0)); /* cqe=10 is arbitrary */

		ofedTDBG(printf("buildContext: calling ibv_req_notify_cq\n");)
		TEST_NZ(ibv_req_notify_cq(s_ctx->cq, 0));
		pthread_mutex_unlock(&buildContextMutex);
		pthread_cond_signal(&buildContextCond);
		ofedTDBG(printf("exiting buildContext\n");)
	}

	bool waitForEvent(rdma_cm_id* id, int eventToWaitFor, ofedContext* s_ctx)
	{
		ofedTDBG(printf("entering waitForEvent\n"));

		rdma_cm_event* event = NULL;
		while (rdma_get_cm_event(id->channel, &event) == 0) {
			struct rdma_cm_event event_copy;
			memcpy(&event_copy, event, sizeof(rdma_cm_event));
			rdma_ack_cm_event(event);

			switch (event_copy.event) {
			case RDMA_CM_EVENT_ADDR_RESOLVED:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_ADDR_RESOLVED\n"));
				eventAddressResolved(event_copy.id, s_ctx);
				break;
			case RDMA_CM_EVENT_ROUTE_RESOLVED:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_ROUTE_RESOLVED\n"));
				eventRouteResolved(event_copy.id);
				break;
			case RDMA_CM_EVENT_CONNECT_REQUEST:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_CONNECT_REQUEST\n"));
				eventConnectionRequested(event_copy.id, s_ctx);
				break;
			case RDMA_CM_EVENT_ESTABLISHED:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_ESTABLISHED\n"));
				eventConnection();
				break;
			case RDMA_CM_EVENT_DISCONNECTED:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_DISCONNECTED\n"));
				eventDisconnection();
				break;
			case RDMA_CM_EVENT_REJECTED:
				ofedTDBG(printf("waitForEvent: RDMA_CM_EVENT_REJECTED (RETRY)\n"));
				return false;
				break;

			default:
				printf("waitForEvent: received event %d\n", (int) event_copy.event);
				die("on_event: unknown event.");
				break;
			}
			ofedTDBG(printf("waitForEvent: loop again\n"));

			if (event_copy.event == eventToWaitFor)
				break;
		}

		ofedTDBG(printf("exiting waitForEvent\n"));
		return true;
	}

	inline bool getCqPollerThreadEnded(int id) {
	 	return cqPollerThreadEnded[id];
	}

	const int getProcId() const { return procId;}

	pthread_t getSendThread() {
		return sendThread;
	}

	void waitSendThread() {
		if (!sendThreadEnded) {
			void* r;
		 	pthread_join(sendThread, &r);
		 	sendThreadEnded = true;
		}
	}

private:

	int eventConnectionRequested(rdma_cm_id* id, ofedContext* s_ctx)
	{
		struct rdma_conn_param cm_params;
		ofedTDBG(printf("entering event_connection_requested\n"));

		ofedConnection* c = searchConnectionById(id);

		buildContext(id->verbs, s_ctx);
		c->buildConnection(id);

		memset(&cm_params, 0, sizeof(rdma_conn_param));

		cm_params.initiator_depth = 1;
		cm_params.responder_resources = 1;
		cm_params.retry_count = 7;
		cm_params.rnr_retry_count = 7;

		TEST_NZ(rdma_accept(id, &cm_params));

		ofedTDBG(printf("exiting event_connection_requested\n"));
		return 0;
	}

	ofedConnection* searchConnectionById(rdma_cm_id* id) {
		for (unsigned int i=0; i<connections.size(); i++) {
			if (connections[i]->id->channel->fd == id->channel->fd) {
				ofedTDBG(printf("searchConnectionById connection %d desc %d\n", i, connections[i]->id->channel->fd));
				return connections[i];
			}
		}
		return NULL;
	}

	inline int eventRouteResolved(rdma_cm_id* id)
	{
		ofedTDBG(printf("entering eventRouteResolved\n"));
		rdma_conn_param cm_params;
		memset(&cm_params, 0, sizeof(rdma_conn_param));

		cm_params.initiator_depth = 1;
		cm_params.responder_resources = 1;
		cm_params.retry_count = 7;
		cm_params.rnr_retry_count = 7;

		TEST_NZ(rdma_connect(id, &cm_params));

		ofedTDBG(printf("exiting eventRouteResolved\n"));
		return 0;
	}

	inline int eventAddressResolved(rdma_cm_id* id, ofedContext* s_ctx)
	{
		ofedTDBG(printf("entering event_address_resolved\n"));
		ofedConnection* c = searchConnectionById(id);
		buildContext(id->verbs, s_ctx);
		c->buildConnection(id);
		// sprintf((char*) s_conn->rdma_local_region, "message from active/client side with pid %d", getpid());
		TEST_NZ(rdma_resolve_route(id, TIMEOUT_IN_MS));

		ofedTDBG(printf("exiting event_address_resolved\n"));
		return 0;
	}

	inline int eventConnection()
	{
		return 0;
	}

	int eventDisconnection()
	{
		return 1;
	}

	static void* sendFromThread(void* param) {
		ofedTDBG(printf("entering sendFromThread\n"));

		ofedTransport* tr = (ofedTransport*) param;
		std::deque<ofedConnection*>* connections = &tr->connections;
		int peers = connections->size();
		ofedTDBG(printf("sendFromThread peers: %d\n", peers));

		bool ending[peers];
		bool ended[peers];
		int endedPeers;
		unsigned int readCounter[peers];
		unsigned int writeCounter[peers];

		for (int i=0; i<peers; i++) {
			ending[i] = false;
			ended[i] = false;
			readCounter[i] = 0;
			writeCounter[i] = 0;
		}
		endedPeers = 0;

		int c=0;
		bool lSleep = true;
		int loops = 0;

		while (true) {
			ofedConnection* th = (*connections)[c];

			ofedTDBG(printf("sendFromThread: loop %d\n", c));

			while (th->getPolledPackets() != th->getSentPackets())
				usleep(1000);

			if (!ended[c]) {
				if (th->isSender()) {
					unsigned int wc = th->getWriteCounter();
					if (wc != writeCounter[c]) {
						th->sendHead(ending[c]);
						writeCounter[c] = wc;
					}

				} else {
					unsigned int rc = th->getReadCounter();
					if (rc != readCounter[c]) {
						th->sendTail();
						readCounter[c] = rc;
					}
				}
			}


			loops++;

			if (ending[c]) {
				ofedTDBG(printf("sendFromThread: QUITTING...\n"));
				ended[c] = true;
				if (!th->isSender()) { 		// We start farm shutdown when receiver get EOS
					th->sendDone(true);
				}

				th->setEnded();

				endedPeers ++;

			} else if (th->isEnding()) {
				ofedTDBG(printf("sendFromThread: Asked to quit... force send\n"));
				ending[c] = true;
			}

			ofedTDBG(printf("sendFromThread: peers: %d, endedPeers: %d\n", peers, endedPeers));
			if (endedPeers == peers)
				break;

			do {
				c++;

				if (c>=peers) {
					ofedTDBG(printf("sendFromThread: beginning from first peer \n"));
					c=0;
					if (lSleep)
						usleep(250);
					else
						lSleep = true;
				}

			} while (ended[c]);
		}

		ofedTDBG(printf("exiting sendFromThread\n"));
		return NULL;
	}

	static void* readCqEvents(void* param)
	{
		readCqThreadParams* p = (readCqThreadParams*) param;
		ofedTransport* th = p->transport;
		ofedContext* s_ctx = p->s_ctx;
		int descId = p->id;
		int peers = p->peers;

		ofedConnection** connections = p->connections;
		int id = 0;

		pthread_mutex_lock(&th->readCqThreadMutex);
		id = th->threadCounter;
		th->threadCounter ++;
		pthread_mutex_unlock(&th->readCqThreadMutex);

		ofedTDBG(printf("entering readCqEvents\n"));

		ofedTDBG(printf("readCqEvents: get events\n"));

		int doneReceived = 0;

		while (true) {
			if (doneReceived == peers) {
				bool bBreak = true;
				for (int i=0; i<peers; i++) {
					if (connections[i]->getSentPackets() != connections[i]->getPolledPackets())
						bBreak = false;
				}

				if (bBreak)
					break;
			}

			ibv_cq* cq;
			void* c = NULL;
			ofedTDBG(printf("readCqEvents: getting cq events\n"));
			TEST_NZ(ibv_get_cq_event(s_ctx->comp_channel, &cq, &c));
			TEST_NZ(ibv_req_notify_cq(cq, 0));
			ibv_ack_cq_events(cq, 1);
			ofedTDBG(printf("readCqEvents: RECEIVED SOME EVENTS\n"));
			int ret = 0;
			ibv_wc wc;

			do {
				ret = ibv_poll_cq(cq, 1, &wc);

				if (ret < 0) die("Failed to poll completions from the CQ\n");

				if (ret == 0) continue;
				ofedConnection* conn = (ofedConnection*) wc.wr_id;

				rdmaMessage* remoteMsg = conn->getRecvMr()->getMemory();
				int connState = conn->getMrRecvState();

				ofedTDBG(printf("readCqEvents: ibv_poll_cq returned %d\n", ret));

				if (wc.status != IBV_WC_SUCCESS) {
					ofedTDBG(printf("STATUS IS %d \n", wc.status));
					printf("STATUS IS %d \n", wc.status);
					die("readCqEvents: status is not IBV_WC_SUCCESS.");
				}

				ofedTDBG(printf("readCqEvents: sender head is %ld.\n", (long) conn->getRdmaSenderMr()->getMemory()->head));
				ofedTDBG(printf("readCqEvents: sender sendMemory is %ld.\n", (long) conn->getRdmaSenderMr()->getMemory()->sentMemory));
				ofedTDBG(printf("readCqEvents: receiver tail is %ld.\n", (long) conn->getRdmaReceiverMr()->getMemory()->tail));

				if (wc.opcode & IBV_WC_RECV) {
					conn->decreaseReceives();
					conn->rdmaRecv();

					if (connState == 0) {
						ofedTDBG(printf("readCqEvents: RECEIVED MEMORY REGISTRATION. desc %d\n", conn->id->channel->fd));
						conn->getSendMr()->setPeerMr(&remoteMsg->data.mr);

						if (conn->isSender())
							conn->getRdmaSenderMr()->setPeerMr(&remoteMsg->data.mr);
						else
							conn->getRdmaReceiverMr()->setPeerMr(&remoteMsg->data.mr);

						conn->setMrRecvState(conn->getMrRecvState()+1);
						if (conn->isActive())
							conn->sendMr();

					} else {
						ofedTDBG(printf("readCqEvents: RECEIVED RDMA MESSAGE.\n"));
						if (conn->getRecvState() != ofedConnection::RS_DONE_RECV) {
							if (conn->getDone()) {
								conn->prepareDisconnection();
								if (conn->isSender())
								 	conn->sendDone(true);

								conn->setRecvState(ofedConnection::RS_DONE_RECV);
								doneReceived ++;
							}
						}
					}
				} else
					conn->increasePolledPackets();

			} while (ret);
		}

		th->cqPollerThreadEnded[descId] = true;

		ofedTDBG(printf("exiting readCqEvents\n"));
		return NULL;
	}
};

} // namespace

#endif /* _OFEDTRANSPORT_HPP_INCLUDE_ */
