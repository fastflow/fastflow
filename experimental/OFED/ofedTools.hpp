#ifndef _FF_OFEDTOOLS_HPP_
#define _FF_OFEDTOOLS_HPP_

/*
 * these parameters are needed to achieve better performance
 * with a sending thread
*/

#include <ff/d/ofedSettings.hpp>

#include <string.h>
#include <rdma/rdma_cma.h>

/* TODO */
#define TEST_NZ(x) do { if ( (x)) die("error: " #x " failed (returned non-zero)." ); } while (0)
#define TEST_Z(x)  do { if (!(x)) die("error: " #x " failed (returned zero/null)."); } while (0)

static void die(const char *reason);

void die(const char *reason)
{
	fprintf(stderr, "%s\n", reason);
	exit(EXIT_FAILURE);
}

#if defined(DEBUG)
#define ofedTDBG(x) x
#else
#define ofedTDBG(x)
#endif

namespace ff {

class msg_t {

private:
	void* m_data;
	size_t m_size;

public:
	typedef void (msg_callback_t) (void *, void *);
	enum {HEADER_LENGHT=4}; // n. of bytes
	/**
	 * Default constructor: create an empty ØMQ message
	 */

	inline msg_t() {
		m_data = NULL;
		m_size=0;
	}

	inline ~msg_t() {
#ifdef MEMCPY_MESSAGES
		free(m_data);
#endif
	}

	/**
	 * Default constructor (2): create a structured ØMQ message.
	 *
	 * \param data the content of the message
	 * \param size memory to be allocated for the message
	 * \param cb   a callback function
	 * \param arg  additional arguments.
	 */
	inline msg_t(const void* data, size_t size, msg_callback_t cb=0, void* arg=0) {
		m_size=size;
#ifdef MEMCPY_MESSAGES
		m_data = malloc(size);
		memcpy(m_data, data, size);
#else
		m_data = const_cast<void*>(data);
#endif
	}


	/**
	 * Initialise the message object (like the constructor but
	 * it rebuilds the object from scratch using new values).
	 */
	inline void init(const void* data, size_t size, msg_callback_t cb=0, void* arg=0) {
		ofedTDBG(printf("entering msg_t::init\n"));
		m_size=size;
#ifdef MEMCPY_MESSAGES
		m_data = malloc(size);
		memcpy(m_data, data, size);
#else
		m_data = const_cast<void*>(data);
#endif
		ofedTDBG(printf("exiting msg_t::init\n"));
	}

	/**
	 * Copy the message
	 */
	inline void copy(msg_t& msg) {
		ofedTDBG(printf("entering msg_t::copy\n"));
		m_data = msg.m_data;
		m_size = msg.m_size;
	}

	/**
	 * Retrieve the message content
	 * \returns a pointer to the content of the message object
	 */
	inline void* getData() {
		return m_data;
	}

	/**
	 * \returns the size in bytes of the content of the message object
	 */
	inline size_t size() {
		return m_size;
	}
};

enum msgType {
	MSG_MR,
	MSG_RDMA,
	MSG_DONE
};

struct rdmaMessage {
	msgType type;
	union {
		struct ibv_mr mr;
	} data;
};

template <class myType>
class registeredMemory {
private:
	myType* memory;
	ibv_mr* memReg;
	ibv_mr* peerMr;

public:
	registeredMemory(ibv_pd* pd, bool remote) {
		memory = new myType();
		peerMr = (ibv_mr*) malloc(sizeof(ibv_mr));

		if (remote)
			TEST_Z(memReg = ibv_reg_mr(pd, memory, sizeof(myType), (ibv_access_flags) (IBV_ACCESS_LOCAL_WRITE | IBV_ACCESS_REMOTE_WRITE)));
		else
			TEST_Z(memReg = ibv_reg_mr(pd, memory, sizeof(myType), IBV_ACCESS_LOCAL_WRITE));
	}

	~registeredMemory() {
		ibv_dereg_mr(memReg);
		delete memory;
		free(peerMr);
	}

	myType* getMemory() {
		return memory;
	}

	void setPeerMr(ibv_mr* mr) {
		memcpy(peerMr, mr, sizeof(ibv_mr));
	}

	ibv_mr* getPeerMr() {
		return peerMr;
	}

	ibv_mr* getMr() {
		return memReg;
	}
};

class workingRequest {
private:
	ibv_send_wr* wr;
	ibv_send_wr* tail;
	int size;
	void* conn;
	static int countInline;

public:
	workingRequest(void* c) {
		wr = NULL;
		tail = NULL;
		size = 0;
		conn = c;
	}

	~workingRequest() {
		ibv_send_wr* next;
		while (wr) {
			 next = wr->next;
			 free(wr->sg_list);
			 free(wr);
			 wr = next;
		}
	}

	template <class myType>
	bool addRDMASend(void* address, size_t size, registeredMemory<myType>* m, bool lForceSignaled = false) {
		struct ibv_send_wr* w = (ibv_send_wr*) malloc(sizeof(ibv_send_wr));
		ibv_sge* sge = (ibv_sge*) malloc(sizeof(ibv_sge));
		bool lInline = true;

		memset(w, 0, sizeof(ibv_send_wr));

		if (!wr)
			wr = w;
		else
			tail->next = w;

		tail = w;

		w->wr_id = (uintptr_t) conn;
		w->opcode = IBV_WR_RDMA_WRITE;
		w->next = NULL;
		w->sg_list = sge;
		w->num_sge = 1;

		if (!lForceSignaled && size < MAX_INLINE_DATA && countInline < 200) {
			w->send_flags = (ibv_send_flags) (IBV_SEND_INLINE | IBV_SEND_SIGNALED);
			countInline++;
		} else {
			w->send_flags = IBV_SEND_SIGNALED;
			countInline = 0;
			lInline = false;
		}

		w->wr.rdma.remote_addr = (uintptr_t)((char*) m->getPeerMr()->addr+ ((char*) address - (char*) m->getMemory()));
		w->wr.rdma.rkey = m->getPeerMr()->rkey;

		sge->addr = (uintptr_t) address;
		sge->length = size;
		sge->lkey = m->getMr()->lkey;

		this->size++;

		return lInline;
	}

	void addSendMsg(registeredMemory<rdmaMessage>* m) {
		struct ibv_send_wr* w = (ibv_send_wr*) malloc(sizeof(ibv_send_wr));
		ibv_sge* sge = (ibv_sge*) malloc(sizeof(ibv_sge));

		memset(w, 0, sizeof(ibv_send_wr));

		if (!wr)
			wr = w;
		else
			tail->next = w;

		tail = w;

		w->wr_id = (uintptr_t)conn;
		w->opcode = IBV_WR_SEND;
		w->next = NULL;
		w->sg_list = sge;
		w->num_sge = 1;
		w->send_flags = IBV_SEND_SIGNALED;

		sge->addr = (uintptr_t) m->getMr()->addr;
		sge->length = sizeof(rdmaMessage);
		sge->lkey = m->getMr()->lkey;
	}

	ibv_send_wr* getRequest() {
		return wr;
	}

	int getSize() {
		return size;
	}
};
int workingRequest::countInline = 0;

}

#endif
