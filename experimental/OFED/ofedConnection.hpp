#include <ff/d/ofedTools.hpp>

#ifndef _FF_OFEDCONNECTION_HPP_
#define _FF_OFEDCONNECTION_HPP_

namespace ff {

class senderSharedMemory {

public:
	char m_data[BUFFER_SIZE];
	unsigned int head;
	unsigned int sentMemory;
	bool done;

	/**
	 * Default constructor: create an empty ØMQ message
	 */
	inline senderSharedMemory() {
		memset(m_data,0,BUFFER_SIZE);
		head = 0;
		sentMemory = 0;
		done = false;
	}
};

class receiverSharedMemory {

public:
	unsigned int tail;
	bool done;
	unsigned int request;

	/**
	 * Default constructor: create an empty ØMQ message
	 */

	inline receiverSharedMemory() {
		tail = 0;
		request = 0;
		done = false;
	}
};

class rdmaQueue {
private:
	senderSharedMemory* senderSharedMemoryPtr;
	receiverSharedMemory* receiverSharedMemoryPtr;

	char* senderSharedMemoryDataPtr;

public:
	inline rdmaQueue(senderSharedMemory* sm, receiverSharedMemory* rm) {
		senderSharedMemoryPtr = sm;
		receiverSharedMemoryPtr = rm;
		senderSharedMemoryDataPtr = sm->m_data;
	}

	/**
	 * Retrieve the message content
	 * \returns a pointer to the content of the message object
	 */
	inline bool enqueue(const msg_t& m) {
		ofedTDBG(printf("entering enqueue\n"));

		msg_t& message = const_cast<msg_t&>(m);
		size_t msgsize = message.size();
		size_t size = sizeof(size_t);
		ofedTDBG(printf("enqueue: size is %d\n", (int) msgsize));

		/*
 		if (((char*) message.getData())[0] == 'E')
 			printf("EOS\n");
 		*/

		unsigned int r = BUFFER_SIZE;

		bool lBeginning = false;
		if (senderSharedMemoryPtr->head < senderSharedMemoryPtr->sentMemory) {
			if (senderSharedMemoryPtr->head+ size+ msgsize >= senderSharedMemoryPtr->sentMemory) {
				ofedTDBG(printf("enqueue: wait for sending messages 1\n"));
				return false;
			}
		}
			if (senderSharedMemoryPtr->head+ size+ msgsize >= r) {
				if (size+ msgsize >= senderSharedMemoryPtr->sentMemory) {
					ofedTDBG(printf("enqueue: wait for sending messages head: %d, sentMemory: %d size: %d, msgsize: %d\n", (int) senderSharedMemoryPtr->head, (int) senderSharedMemoryPtr->sentMemory, (int) size, (int) msgsize));
					return false;
				}
		}

		if (senderSharedMemoryPtr->head < receiverSharedMemoryPtr->tail) {
			if (senderSharedMemoryPtr->head+ size+ msgsize >= receiverSharedMemoryPtr->tail) {
				ofedTDBG(printf("enqueue: return false because head < tail and queue full\n"));
				return false;
			}
		} else {
			if (senderSharedMemoryPtr->head+ size+ msgsize >= r) {
				if (size+ msgsize>=receiverSharedMemoryPtr->tail) {
					ofedTDBG(printf("enqueue: return false because head > tail and queue full\n"));

					return false;
				}
				lBeginning = true;
			}
		}

		char* pointer;

		if (lBeginning) {
			ofedTDBG(printf("enqueue: begin again\n"));
			pointer = senderSharedMemoryPtr->m_data+senderSharedMemoryPtr->head;
			*((size_t*)pointer) = 0;
			ofedTDBG(printf("enqueue: setting to 0: %d \n", senderSharedMemoryPtr->head));
			senderSharedMemoryPtr->head = 0;
		}

		pointer = senderSharedMemoryPtr->m_data+senderSharedMemoryPtr->head;

		*((size_t*)pointer) = msgsize;
		pointer += size;

		memcpy(pointer, message.getData(), msgsize);
		senderSharedMemoryPtr->head += (int) msgsize+ size;

		ofedTDBG(printf("enqueue: new head is %d.\n", senderSharedMemoryPtr->head));
		ofedTDBG(printf("enqueue: new tail is %d.\n", receiverSharedMemoryPtr->tail));

		ofedTDBG(printf("exiting enqueue\n"));
		return true;
	}

	inline bool tryDequeue() {
		if (receiverSharedMemoryPtr->tail == senderSharedMemoryPtr->sentMemory) {
			return false;
		}

		return true;
	}

	/**
	 * Retrieve the message content
	 * \returns a pointer to the content of the message object
	 */
	inline bool dequeue(msg_t& m) {
		ofedTDBG(printf("entering dequeue\n"));
		ofedTDBG(printf("dequeue: current tail is     %d\n", (int) receiverSharedMemoryPtr->tail));
		ofedTDBG(printf("dequeue: current lastSent is %d\n", (int) senderSharedMemoryPtr->sentMemory));
		size_t msgsize = 0;
		size_t size = sizeof(size_t);

		if (receiverSharedMemoryPtr->tail == senderSharedMemoryPtr->sentMemory) {
			return false;
		}

		char* pointer = senderSharedMemoryDataPtr+ receiverSharedMemoryPtr->tail;
		msgsize = *(size_t*)(pointer);
		if ((unsigned int) msgsize == 0) {
			ofedTDBG(printf("dequeue: reset tail\n"));
			receiverSharedMemoryPtr->tail = 0;
			pointer = senderSharedMemoryDataPtr;

			msgsize = *(size_t*)(pointer);
		}

		pointer += size;
		ofedTDBG(printf("dequeue: size is %d\n", (int) msgsize));

		m.init(pointer, msgsize);

		/*
 		if (((char*) m.getData())[0] == 'E')
 			printf("EOS\n");
 		*/

		receiverSharedMemoryPtr->tail += msgsize+ size;

		ofedTDBG(printf("dequeue: new tail is %d\n", (int) receiverSharedMemoryPtr->tail));

		ofedTDBG(printf("exiting dequeue\n"));
		return true;
	}

	inline void sendRequest() {
		receiverSharedMemoryPtr->request ++;
	}

	inline unsigned int getRequest() {
		return receiverSharedMemoryPtr->request;
	}

	inline void getRequestPtr(unsigned int** address, size_t* size) {
		*address = (unsigned int*) (&receiverSharedMemoryPtr->request);
		*size = sizeof(unsigned int);
	}

	inline void getToBeSentPtr(void** address, size_t* size) {
		*address = senderSharedMemoryPtr->m_data+ senderSharedMemoryPtr->sentMemory;
		*size = *(size_t*)(*address);
	}

	inline void updateSentMemory(unsigned int sent) {
		senderSharedMemoryPtr->sentMemory = sent;
	}

	inline void getLastSentPtr(unsigned int** address, size_t* size) {
		*address = (unsigned int*) &senderSharedMemoryPtr->sentMemory;
		*size = sizeof(unsigned int);
	}

	inline void getHeadPtr(unsigned int** address, size_t* size) {
		*address = (unsigned int*) &senderSharedMemoryPtr->head;
		*size = sizeof(unsigned int);
	}

	inline void getTailPtr(unsigned int** address, size_t* size) {
		*address = (unsigned int*) &receiverSharedMemoryPtr->tail;
		*size = sizeof(unsigned int);
	}

	inline void getSenderDonePtr(bool** address, size_t* size) {
		*address = &senderSharedMemoryPtr->done;
		*size = sizeof(bool);
	}

	inline void getReceiverDonePtr(bool** address, size_t* size) {
		*address = &receiverSharedMemoryPtr->done;
		*size = sizeof(bool);
	}

	inline void* getPtr(unsigned int position) {
		return senderSharedMemoryPtr->m_data+ position;
	}
};

struct ofedContext {
	ibv_context *ctx;
	ibv_pd *pd;
	ibv_cq *cq;
	ibv_comp_channel *comp_channel;
};

class ofedConnection {

public:
	enum recv_states {
		RS_INIT,
		RS_MR_RECV,
		RS_DONE_RECV,
	};

	enum send_states {
		SS_INIT,
		SS_MR_SENT,
		SS_RDMA_SENT,
		SS_DONE_SENT,
	};


private:
	ibv_qp *qp;
	ofedContext* s_ctx;

	registeredMemory<rdmaMessage>* remoteMr;
	registeredMemory<rdmaMessage>* localMr;
	registeredMemory<senderSharedMemory>* rdmaSenderMr;
	registeredMemory<receiverSharedMemory>* rdmaReceiverMr;

	rdmaQueue* rdmaSharedQueue;

	int send_state;
	int mr_send_state;

	int recv_state;
	int mr_recv_state;

	int receives;
	unsigned int readCounter;
	unsigned int writeCounter;
	unsigned int reqCounter;
	bool ending;
	bool ended;
	bool active;
	bool sender;
	unsigned long polledPackets;
	unsigned long sentPackets;

public:
	rdma_cm_id *id;
	ofedConnection(ofedContext* ctx, bool active, bool sender):
		qp(NULL), s_ctx(NULL), remoteMr(NULL), localMr(NULL), rdmaSenderMr(NULL), rdmaReceiverMr(NULL),
		rdmaSharedQueue(NULL), send_state(0), mr_send_state(0), recv_state(0), mr_recv_state(0), id(NULL) {

		s_ctx = ctx;
		receives = 0;
		readCounter = 0;
		writeCounter = 0;
		reqCounter = 0;
		ending = false;
		ended = false;
		this->active = active;
		this->sender = sender;
		polledPackets = 0;
		sentPackets = 0;
	}

	~ofedConnection() {
	}

	inline void rdmaSend(workingRequest* w) {
		ofedTDBG(printf("entering rdmaSend\n"));

		struct ibv_send_wr* wr, *bad_wr;
		wr = w->getRequest();

		int ibvPostSend = ibv_post_send(qp, wr, &bad_wr);
		if (ibvPostSend != 0)
			printf("ibv_post_send returned %d, errno %d\n", ibvPostSend, errno);
		ofedTDBG(printf("ibv_post_send returned %d, errno %d\n", ibvPostSend, errno));
		TEST_NZ(ibvPostSend);

		ofedTDBG(printf("exiting rdmaSend\n"));
	}

	inline void rdmaRecv(int num = NUMBER_OF_POST_RECV)
	{
		ofedTDBG(printf("entering rdmaRecv\n"));
		int x = receives;
		ofedTDBG(printf("rdmaRecv... before counter: %d\n", receives));
		for (int i=x; i<num; i++) {
			postReceives();
			ofedTDBG(printf("rdmaRecv... counter now is: %d\n", receives));
			receives++;
		}
		ofedTDBG(printf("rdmaRecv... after counter: %d\n", receives));
		ofedTDBG(printf("exiting rdmaRecv\n"));
	}

public:
	inline registeredMemory<rdmaMessage>* getRecvMr() {
		return remoteMr;
	}

	inline registeredMemory<rdmaMessage>* getSendMr() {
		return localMr;
	}

	inline registeredMemory<senderSharedMemory>* getRdmaSenderMr() {
		return rdmaSenderMr;
	}

	inline registeredMemory<receiverSharedMemory>* getRdmaReceiverMr() {
		return rdmaReceiverMr;
	}

	inline bool getDone() {
		bool done;

		size_t size;
		bool* pointer;
		if (sender)
			rdmaSharedQueue->getReceiverDonePtr(&pointer, &size);
		else
			rdmaSharedQueue->getSenderDonePtr(&pointer, &size);
		done = *pointer;

		return done;
	}

	inline int getRecvState() {
		return recv_state;
	}

	inline void setRecvState(int s) {
		recv_state = s;
	}

	inline int getMrRecvState() {
		return mr_recv_state;
	}

	inline void setMrRecvState(int s) {
		mr_recv_state = s;
	}

	inline int getSendState() {
		return send_state;
	}

	inline void decreaseReceives() {
		receives --;
	}

	inline int getMrSendState() {
		return mr_send_state;
	}

	inline bool isEnding() {
		return ending;
	}

	inline void setEnded() {
		ended = true;
	}

	inline bool isEnded() {
		return ended;
	}

	inline void setMrSendState(int s) {
		mr_send_state = s;
	}

	inline rdmaQueue* getRdmaSharedQueue() {
		return rdmaSharedQueue;
	}


private:
	void postReceives()
	{
		ofedTDBG(printf("entering postReceives\n"));
		struct ibv_recv_wr wr, *bad_wr = NULL;
		struct ibv_sge sge;

		wr.wr_id = (uintptr_t)this;
		wr.next = NULL;
		wr.sg_list = &sge;
		wr.num_sge = 1;

		sge.addr = (uintptr_t)remoteMr->getMemory();
		sge.length = sizeof(rdmaMessage);
		sge.lkey = remoteMr->getMr()->lkey;

		int ibvPostRecv = ibv_post_recv(qp, &wr, &bad_wr);
		ofedTDBG(printf("postReceives ibv_post_recv returned %d\n", errno));
		TEST_NZ(ibvPostRecv);
		ofedTDBG(printf("exiting postReceives\n"));
	}

	inline void sendMsg()
	{
		ofedTDBG(printf("entering sendMsg\n"));
		struct ibv_send_wr wr, *bad_wr = NULL;
		struct ibv_sge sge;

		memset(&wr, 0, sizeof(ibv_send_wr));

		wr.wr_id = (uintptr_t)this;
		wr.opcode = IBV_WR_SEND;
		wr.sg_list = &sge;
		wr.num_sge = 1;
		wr.send_flags = IBV_SEND_SIGNALED;

		sge.addr = (uintptr_t)localMr->getMemory();
		sge.length = sizeof(rdmaMessage);
		sge.lkey = localMr->getMr()->lkey;

		TEST_NZ(ibv_post_send(qp, &wr, &bad_wr));
		ofedTDBG(printf("exiting sendMsg\n"));
	}

	void waitMr() {
		ofedTDBG(printf("entering waitMr %d\n", (int)this->id->channel->fd));

		while (mr_recv_state < RS_MR_RECV)
			usleep(1000);

		ofedTDBG(printf("waitMr: MEMORY REGISTRATION COMPLETED\n"));

		ofedTDBG(printf("exiting waitMr\n"));
	}

public:
	void sendMr() {
		ofedTDBG(printf("entering sendMr %d\n", (int)this->id->channel->fd));
		localMr->getMemory()->type = MSG_MR;

		if (sender)
			memcpy(&localMr->getMemory()->data.mr, rdmaReceiverMr->getMr(), sizeof(ibv_mr));
		else
			memcpy(&localMr->getMemory()->data.mr, rdmaSenderMr->getMr(), sizeof(ibv_mr));

		sendMsg();
		sentPackets += 1;

		ofedTDBG(printf("exiting sendMr\n"));
	}

	void initConnection() {
		ofedTDBG(printf("entering initConnection: %d, transortMsg_t: %d\n", (int) sizeof(rdmaMessage), (int) sizeof(senderSharedMemory)));

		localMr = new registeredMemory<rdmaMessage>(s_ctx->pd, false);
		remoteMr = new registeredMemory<rdmaMessage>(s_ctx->pd, true);

		if (sender) {
			rdmaSenderMr = new registeredMemory<senderSharedMemory>(s_ctx->pd, false);
			rdmaReceiverMr = new registeredMemory<receiverSharedMemory>(s_ctx->pd, true);
		} else {
			rdmaSenderMr = new registeredMemory<senderSharedMemory>(s_ctx->pd, true);
			rdmaReceiverMr = new registeredMemory<receiverSharedMemory>(s_ctx->pd, false);
		}
		rdmaSharedQueue = new rdmaQueue(rdmaSenderMr->getMemory(), rdmaReceiverMr->getMemory());

		ofedTDBG(printf("exiting initConnection\n"));
	}

	void registerMemory() {
		ofedTDBG(printf("entering registerMemory\n"));
		if (!active)
			sendMr();

		ofedTDBG(printf("registerMemory: sentMr\n"));
		waitMr();
		sendDone(false);

		ofedTDBG(printf("exiting registerMemory\n"));
	}

	void buildConnection(rdma_cm_id* id)
	{
		ofedTDBG(printf("entering buildConnection\n"));
		ibv_qp_init_attr qp_attr;

		memset(&qp_attr, 0, sizeof(ibv_qp_init_attr));

		qp_attr.send_cq = s_ctx->cq;
		qp_attr.recv_cq = s_ctx->cq;
		qp_attr.qp_type = IBV_QPT_RC;

		qp_attr.cap.max_send_wr = MAX_SEND_WR;
		qp_attr.cap.max_recv_wr = MAX_RECV_WR;
		qp_attr.cap.max_send_sge = 1;
		qp_attr.cap.max_recv_sge = 1;
		qp_attr.cap.max_inline_data = MAX_INLINE_DATA;

		TEST_NZ(rdma_create_qp(id, s_ctx->pd, &qp_attr));

		id->context = this;

		this->id = id;
		qp = id->qp;

		mr_send_state = SS_INIT;
		mr_recv_state = RS_INIT;

		send_state = SS_INIT;
		recv_state = RS_INIT;

		ofedTDBG(printf("exiting buildConnection\n"));
	}

	void destroyConnection()
	{
		ofedTDBG(printf("entering destroy_connection\n"));

		rdma_destroy_qp(id);

		delete localMr;
		delete remoteMr;
		delete rdmaSenderMr;
		delete rdmaReceiverMr;

		delete rdmaSharedQueue;

		rdma_event_channel *ec = id->channel;
		rdma_destroy_id(id);
		rdma_destroy_event_channel(ec);

		ofedTDBG(printf("exiting destroy_connection\n"));
	}

	inline void sendDone(bool done) {
		ofedTDBG(printf("entering sendDone\n"));

		localMr->getMemory()->type = MSG_RDMA;
		workingRequest* w = new workingRequest((void*) this);

		size_t size;
		bool* pointer;
		if (sender)
			rdmaSharedQueue->getSenderDonePtr(&pointer, &size);
		else
			rdmaSharedQueue->getReceiverDonePtr(&pointer, &size);
		(*pointer) = done;

		if (sender)
			w->addRDMASend((void*) pointer, size, rdmaSenderMr, true);
		else
			w->addRDMASend((void*) pointer, size, rdmaReceiverMr, true);

		w->addSendMsg(getSendMr());
		rdmaSend(w);
		delete(w);

		sentPackets += 2;

		ofedTDBG(printf("exiting sendDone\n"));
	}

	inline void sendTail() {
		ofedTDBG(printf("entering sendTail\n"));

		workingRequest* w = new workingRequest((void*) this);
		localMr->getMemory()->type = MSG_RDMA;

		size_t size;
		unsigned int* pointer;
		rdmaSharedQueue->getTailPtr(&pointer, &size);

		w->addRDMASend((void*) pointer, size, rdmaReceiverMr, true);
		w->addSendMsg(getSendMr());
		rdmaSend(w);
		delete(w);

		sentPackets += 2;

		ofedTDBG(printf("exiting sendTail\n"));
	}

	inline void sendHead(bool bForce) {
		ofedTDBG(printf("entering sendHead\n"));

		bool bEvent = false;
		int nEvents = 0;

		localMr->getMemory()->type = MSG_RDMA;

		unsigned int* headP;
		size_t headSize;
		rdmaSharedQueue->getHeadPtr(&headP, &headSize);

		unsigned int* lastSentP;
		size_t lastSentSize;
		rdmaSharedQueue->getLastSentPtr(&lastSentP, &lastSentSize);

		unsigned int lastSent = *lastSentP;
		unsigned int head = *headP;
		workingRequest* w = new workingRequest((void*) this);

		if (lastSent > head) {
			ofedTDBG(printf("sendHead: lastSent %d > head %d\n", (int) lastSent, (int) head));
			size_t size;
			void* pointer;
			pointer = rdmaSharedQueue->getPtr(lastSent);
			size = BUFFER_SIZE - lastSent;

			if (size != 0) {
				ofedTDBG(printf("sendHead: sending address: %d size: %d\n", *(unsigned int*) pointer, (int) size));
				w->addRDMASend(pointer, size, rdmaSenderMr, true);
				nEvents ++;
				bEvent = true;
			}

			lastSent = 0;
			*lastSentP = 0;

		}

		if (lastSent < head) {
			ofedTDBG(printf("sendHead: lastSent %d < head %d\n", (int) lastSent, (int) head));
			size_t size;
			void* pointer;
			rdmaSharedQueue->getToBeSentPtr(&pointer, &size);
			size = head- lastSent;

			if (size != 0) {
				ofedTDBG(printf("sendHead: sending address: %d size: %d\n", *(unsigned int*) pointer, (int) size));
				w->addRDMASend(pointer, size, rdmaSenderMr, true);
				nEvents ++;
				bEvent = true;
			}

			lastSent += size;
		}

		if (bEvent) {
			*lastSentP = lastSent;
			w->addRDMASend((void*) lastSentP, lastSentSize, rdmaSenderMr, true);
			w->addSendMsg(getSendMr());

			nEvents += 2;
		}
		rdmaSend(w);
		delete(w);
		sentPackets += nEvents;

		ofedTDBG(printf("exiting sendHead\n"));
		// return bEvent;
	}

	inline void waitDone() {
		while (recv_state < RS_DONE_RECV)
			usleep(1000);
	}

	void prepareDisconnection() {
		ofedTDBG(printf("entering prepareDisconnection\n"));
		if (ended)
			return;

		ending = true;
		ofedTDBG(printf("exiting prepareDisconnection\n"));
	}

	void disconnect() {
		ofedTDBG(printf("entering disconnect\n"));
		waitDone();
		ofedTDBG(printf("exiting disconnect\n"));
	}

	inline bool isActive() {
		return active;
	}

	inline bool isSender() {
		return sender;
	}

	inline bool send(const msg_t& msg) {
		ofedTDBG(printf("entering send %d\n", (int)this->id->channel->fd));
		ofedTDBG(printf("send: try enqueue\n"));
		ofedTDBG(printf("send: sending %d\n", *((int*) const_cast<msg_t&>(msg).getData())));

		while (!rdmaSharedQueue->enqueue(msg)) {
			ofedTDBG(printf("send: enqueue unsuccessful, retrying...\n"));
			usleep(250);
		}
		writeCounter ++;

		ofedTDBG(printf("exiting send\n"));
		return true;
	}

	inline bool recv(msg_t& msg) {
		ofedTDBG(printf("entering recv %d\n", (int)this->id->channel->fd));

		while (true) {
			ofedTDBG(printf("recv: try dequeue\n"));
			if (rdmaSharedQueue->dequeue(msg))
				break;

			ofedTDBG(printf("recv: dequeue unsuccessful, retrying\n"));
			usleep(250);
		}
		readCounter ++;

		ofedTDBG(printf("recv: received %d\n", *((int*) msg.getData())));
		ofedTDBG(printf("exiting recv\n"));
		return true;
	}

	inline bool sendReq() {
		ofedTDBG(printf("entering sendReq %d\n", (int)this->id->channel->fd));

		rdmaSharedQueue->sendRequest();

		unsigned int* requestP;
		size_t reqSize;
		rdmaSharedQueue->getRequestPtr(&requestP, &reqSize);

		workingRequest* w = new workingRequest((void*) this);
		localMr->getMemory()->type = MSG_RDMA;

		w->addRDMASend(requestP, reqSize, rdmaReceiverMr, true);
		w->addSendMsg(getSendMr());
		rdmaSend(w);
		delete(w);

		sentPackets += 2;

		ofedTDBG(printf("exiting sendReq\n"));
		return true;
	}

	inline bool recvReq() {
		ofedTDBG(printf("entering recvReq %d\n", (int)this->id->channel->fd));

		while (true) {
			ofedTDBG(printf("recvReq: try dequeue\n"));
			if (rdmaSharedQueue->getRequest() != reqCounter || ended)
				break;

			ofedTDBG(printf("recvReq: no requests received yet\n"));
			usleep(1000);
		}

		reqCounter ++;

		ofedTDBG(printf("exiting recvReq"));
		return true;
	}

	inline bool tryRecvReq() {
		ofedTDBG(printf("entering tryRecvReq %d\n", (int)this->id->channel->fd));

		ofedTDBG(printf("exiting tryRecvReq"));
		return rdmaSharedQueue->getRequest() != reqCounter;
	}

	inline bool tryRecv() {
		return rdmaSharedQueue->tryDequeue();
	}

	inline unsigned int getReadCounter() {
		return readCounter;
	}

	inline unsigned int getWriteCounter() {
		return writeCounter;
	}

	inline ofedContext* getContext() {
		return s_ctx;
	}

	inline unsigned long getPolledPackets() {
		return polledPackets;
	}

	inline unsigned long getSentPackets() {
		return sentPackets;
	}

	inline void increasePolledPackets() {
		polledPackets ++;
	}

};
}
#endif /* _OFEDCONNECTION_HPP_INCLUDE_ */
