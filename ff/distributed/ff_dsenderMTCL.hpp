/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */
/* Authors:
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FF_DSENDER_H
#define FF_DSENDER_H

#include <iostream>
#include <map>

#include <arpa/inet.h>
#include <cmath>
#include <deque>
#include <ff/distributed/ff_messageAllocator.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/ff.hpp>
#include <netdb.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <sys/uio.h>
#include <sys/un.h>
#include <thread>

#include <cereal/archives/portable_binary.hpp>
#include <cereal/cereal.hpp>
#include <cereal/types/polymorphic.hpp>
#include <cereal/types/vector.hpp>

#include "MTCL/include/mtcl.hpp"

#ifdef RANDOM_LOADBALANCING
#include <random>
#endif

#define MAX_PREALLOC_BUFFERS 4

namespace ff {

class ff_dsenderMTCL2;

struct uBuffer_buf {
	char *payloads;
	char *headers;
	MTCL::Request r;
};

struct uBufferAllocator {
	static inline std::deque<uBuffer_buf *> q;
	static inline size_t payloadsSize = 0, headersSize = 0;

	static inline void init(size_t preAlloc, size_t payloadsSize, size_t headers) {
		uBufferAllocator::payloadsSize = payloadsSize;
		uBufferAllocator::headersSize = headers * headerSize;

		for (size_t i = 0; i < preAlloc; i++) {
			uBuffer_buf *b = new uBuffer_buf;
			b->payloads = (char *)malloc(uBufferAllocator::payloadsSize);
			b->headers = (char *)malloc(uBufferAllocator::headersSize);
			q.push_back(b);
		}
	}

	static inline uBuffer_buf *alloc() {
		if (q.empty()) {
			uBuffer_buf *b = new uBuffer_buf;
			b->payloads = (char *)malloc(uBufferAllocator::payloadsSize);
			b->headers = (char *)malloc(uBufferAllocator::headersSize);
			return b;
		}

		while (true)
			for (size_t i = 0; i < q.size(); i++) {
				uBuffer_buf *b = q.at(i);
				if (MTCL::test(b->r)) {
					q.erase(q.begin() + i);
					return b;
				}
			}
	}

	static inline void dealloc(uBuffer_buf *b) {
		q.push_front(b);
	}

	static inline void clean() {
		while (!q.empty())
			for (size_t i = 0; i < q.size(); i++) {
				uBuffer_buf *b = q.at(i);
				if (MTCL::test(b->r)) {
					free(b->headers);
					free(b->payloads);
					delete b;
					q.erase(q.begin() + i);
				}
			}
	}
};

class uBuffer {
	friend class ff_dsenderMTCL2;
	size_t batchingLimit, sizeLimit;
	unsigned int actualSize = 0;
	ff_dsenderMTCL2 *parent;
	MTCL::HandleUser &handle;
	int &counter;

	uBuffer_buf *buffers = nullptr;

	char *mainBuffer_curr_ptr = nullptr;

	template <typename T>
	static inline T HostToNetworkByteOrder(T value) {
		static_assert(std::is_integral<T>::value, "T must be an integral type.");

		if constexpr (sizeof(T) == 2) // 16-bit values
			return htons(static_cast<uint16_t>(value));
		else if constexpr (sizeof(T) == 4) // 32-bit values
			return htonl(static_cast<uint32_t>(value));
		else if constexpr (sizeof(T) == 8) // 64-bit values
			return htobe64(static_cast<uint64_t>(value));
		else
			return value;
	}

	inline static void pushHeader_(char *buffAddr, addr_t dest, addr_t src, ChannelType t, sizeDFF_t size_) {
		*reinterpret_cast<addr_t *>(buffAddr) = HostToNetworkByteOrder(dest);
		*reinterpret_cast<addr_t *>(buffAddr + sizeof(addr_t)) = HostToNetworkByteOrder(src);
		*reinterpret_cast<ChannelType *>(buffAddr + 2 * sizeof(addr_t)) = t;
		*reinterpret_cast<sizeDFF_t *>(buffAddr + 2 * sizeof(addr_t) + sizeof(ChannelType)) = HostToNetworkByteOrder(size_);
	}

  public:
	uBuffer() = default;

	uBuffer(size_t batchingLimit, size_t sizeLimit, ff_dsenderMTCL2 *parent, MTCL::HandleUser &handle, int &counter) : batchingLimit(batchingLimit), sizeLimit(sizeLimit), parent(parent), handle(handle), counter(counter) {}

	int push(message2_t *);
	int flush(bool = false);
};

class ff_dsenderMTCL2 : public ff_minode_t<message2_t> {
	friend class uBuffer;

	const std::vector<std::tuple<std::string, std::string, std::vector<ff_node *>>> &destEndpoints;
	const std::unordered_map<ff_node *, IngressEgressChannels_t> &channelsDictionary;

  protected:
	size_t neos = 0;
	size_t totalAcks = 0;
	std::unordered_map<int, size_t> dest2ConnID;
	std::vector<MTCL::HandleUser> MTCL_Handlers;
	std::unordered_map<int, std::set<size_t>> src2PossibleDestConnID;
	std::vector<uBuffer> batchBuffers;
	std::vector<int> handlerCounters;
	int batchSize;
	size_t batchByteSize;
	int messageOTF;
	size_t rr_connID;
#ifdef RANDOM_LOADBALANCING
	std::mt19937 gen{std::random_device{}()};
#endif

	int waitAckFromAny() {
		static size_t lastConnId = 0;
		while (true)
			for (size_t j = 0; j < handlerCounters.size(); ++j) {
				size_t i = (lastConnId + j) % handlerCounters.size();
				ssize_t r;
				ack_t a;
				size_t sz;
				if ((r = MTCL_Handlers[i].probe(sz, handlerCounters.size() == 1)) <= 0) { // get blocking only if i need to receive from one handle
					if (errno == EWOULDBLOCK) {
						assert(r == -1);
						continue;
					}
					perror("recvnnb ack");
					return -1;
				} else {
					if (MTCL_Handlers[i].receive(&a, sizeof(ack_t)) == sizeof(ack_t)) {
						handlerCounters[i]++;
						lastConnId = i + 1;
						return i;
					} else
						perror("Receive ack in WaitAckFromAny");
				}
			}
		return -1;
	}

	int waitAckFrom(int &counter) {
		while (!counter) {
			for (size_t i = 0; i < handlerCounters.size(); ++i) {
				ssize_t r;
				ack_t a;
				size_t sz;
				if ((r = MTCL_Handlers[i].probe(sz, handlerCounters.size() == 1)) <= 0) { // get blocking only if i need to receive from one handle
					if (errno == EWOULDBLOCK) {
						assert(r == -1);
						continue;
					}
					perror("recvnnb ack");
					return -1;
				} else {
					if (MTCL_Handlers[i].receive(&a, sizeof(ack_t)) == sizeof(ack_t))
						handlerCounters[i]++;
					else
						perror("Receive ack in WaitAckFrom");
				}
			}

			// TODO: FIX with better way to pause from receving acks
		}
		return 0;
	}

	inline uBuffer &getMostFilledBuffer(const int &src) {
		auto &possibleDestConnID = src2PossibleDestConnID[src];

		if (batchSize > 1) {
			uBuffer *maxBuffer = nullptr;
			for (auto &connID : possibleDestConnID) {
				auto *buffer = &batchBuffers[connID];
				if (!maxBuffer)
					maxBuffer = buffer;
				else if (maxBuffer->actualSize < buffer->actualSize)
					maxBuffer = buffer;
			}

			if (maxBuffer->actualSize > 0)
				return *maxBuffer;
		}

#ifdef RANDOM_LOADBALANCING
		size_t connID;
		std::sample(possibleDestConnID.begin(), possibleDestConnID.end(), &connID, 1, gen);
		return batchBuffers[connID];
#else
		rr_connID = (rr_connID + 1) % possibleDestConnID.size();
		size_t connId = *std::next(possibleDestConnID.begin(), rr_connID);
		if (handlerCounters[connId])
			return batchBuffers[connId];

		connId = waitAckFromAny();
		return batchBuffers[connId];
#endif
	}

  public:
	ff_dsenderMTCL2(const std::vector<std::tuple<std::string, std::string, std::vector<ff_node *>>> &destEndpoints, const std::unordered_map<ff_node *, IngressEgressChannels_t> &channelsDictionary, int batchSize = DEFAULT_BATCH_SIZE, size_t batchByteSize = DEFAULT_BATCH_BYTE_SIZE, int messageOTF = DEFAULT_MESSAGE_OTF) : destEndpoints(destEndpoints), channelsDictionary(channelsDictionary), batchSize(batchSize), batchByteSize(batchByteSize), messageOTF(messageOTF) {}

	int svc_init() {

		// important for the next lines since they are using the reference to the back of the vector
		MTCL_Handlers.reserve(destEndpoints.size());
		handlerCounters.reserve(destEndpoints.size());

		uBufferAllocator::init(
			SEND_BUFFERS_PREALLOCATE,
			2 * std::max<size_t>(batchByteSize, SINGLE_SEND_SIZE_THRESHOLD) + batchSize * headerSize + sizeof(int), // TODO check if is possible to allocate less bytes
			batchSize);

		for (auto &[_, endpoint, dest_v] : destEndpoints) {

			auto h = MTCL::Manager::connect(endpoint, MAX_RETRIES, 10);
			if (!h.isValid()) {
				std::cerr << "Connection failed with this endpoint: " << endpoint << std::endl;
				return -1;
			}

			MTCL_Handlers.push_back(std::move(h));
			handlerCounters.push_back(messageOTF);
			size_t connID = MTCL_Handlers.size() - 1;
			totalAcks += messageOTF;

			batchBuffers.emplace_back(batchSize, batchByteSize, this, MTCL_Handlers[connID], handlerCounters[connID]);

			for (ff_node *n : dest_v)
				dest2ConnID[n->mioID] = connID;
		}

		// build the map for messages without destination, in order to perform the load balacing
		for (auto &[n, channelsTuple] : channelsDictionary)
			for (auto &[dest_n, t, l, q] : channelsTuple.second)
				if (l == ChannelLocality::REMOTE)
					src2PossibleDestConnID[n->mioID].insert(dest2ConnID[dest_n->mioID]);

		// initialize the round robin variabile with the number of all available endpoints, so that probably we will start from zero at the first iteration
		rr_connID = destEndpoints.size();

		return 0;
	}

	message2_t *svc(message2_t *task) {
		if (task->dest >= 0)
			batchBuffers[dest2ConnID[task->dest]].push(task);
		else {
			if (task->isFlush()) {
				for (auto &connID_ : std::unordered_set<int>(src2PossibleDestConnID[task->src].begin(), src2PossibleDestConnID[task->src].end()))
					batchBuffers[connID_].flush();
				MessageAllocator::releaseMessage(task);
			} else if (task->isLogicalEOS()) {

				for (auto &connID_ : src2PossibleDestConnID[task->src]) {
					batchBuffers[connID_].push(MessageAllocator::make_logical_EOS(task->src));
				}

				MessageAllocator::releaseMessage(task);
			} else
				getMostFilledBuffer(task->src).push(task); // get the most filled buffer socket or a rr socket
		}

		return this->GO_ON;
	}

	void svc_end() {
		for (auto &h : MTCL_Handlers)
			h.close();

		size_t currentack = 0;
		for (const auto &counter : handlerCounters)
			currentack += counter;

		ack_t a;
		ssize_t r;
		size_t sz;
		while (currentack < this->totalAcks) {

			for (size_t i = 0; i < MTCL_Handlers.size(); i++) {
				if (handlerCounters[i] == -1)
					continue;
				if ((r = MTCL_Handlers[i].probe(sz, false)) <= 0) {
					if (r == -1 && errno == EWOULDBLOCK) {
						continue;
					}
				} else {
					if (MTCL_Handlers[i].receive(&a, sizeof(ack_t)) == sizeof(ack_t)) {
						currentack++;
						handlerCounters[i]++;
						continue;
					}
				}
				currentack += messageOTF - handlerCounters[i]; // in case of errors consider the lost acks as received
				handlerCounters[i] = -1;
			}
		}
	}

	~ff_dsenderMTCL2() {
		uBufferAllocator::clean();
	}
};

inline int uBuffer::push(message2_t *m) {
	if (!this->buffers) {
		this->buffers = uBufferAllocator::alloc();
		mainBuffer_curr_ptr = this->buffers->payloads;
	}

	if (batchingLimit == 1 || m->size >= sizeLimit) {
		// if there is already something in the buffer FLUSH it to preserve the ordering
		if (actualSize)
			this->flush(true);

		// direct send
		sizeDFF_t effectiveSize = m->size;
		if (effectiveSize > SINGLE_SEND_SIZE_THRESHOLD)
			m->size = 0;

		size_t totalSize = m->size + headerSize + sizeof(int);

		if (m->data && m->size)
			std::memcpy(buffers->payloads, m->data, m->size);

		pushHeader_(buffers->payloads + m->size, m->dest, m->src, m->type, effectiveSize);
		*reinterpret_cast<int *>(buffers->payloads + totalSize - sizeof(int)) = htonl(1);

#ifdef DFF_ACK
		if (counter == 0 && parent->waitAckFrom(counter) == -1) {
			error("Error waiting ack from socket inside the callback\n");
			return -1;
		}
#endif

		if (handle.isend(buffers->payloads, totalSize, buffers->r) < 0) {
			error("Error sending\n");
			return -1;
		}

		if (effectiveSize > SINGLE_SEND_SIZE_THRESHOLD)
			if (handle.send(m->data, effectiveSize) < 0) {
				error("Error sending big payload\n");
				return -1;
			}

		MessageAllocator::releaseMessage(m);
		uBufferAllocator::dealloc(this->buffers);
		this->buffers = nullptr;
		this->mainBuffer_curr_ptr = nullptr;

#ifdef DFF_ACK
		counter -= 1;
#endif
	} else {
		pushHeader_(buffers->headers + actualSize * headerSize, m->dest, m->src, m->type, m->size);
		++actualSize;
		if (m->data && m->size) {
			memcpy(mainBuffer_curr_ptr, m->data, m->size);
			mainBuffer_curr_ptr += m->size;
		}

		// if the entries reached the batching limit OR the size of the buffer reached the sizeLimit OR the message has not size (i.e. is a logical EOS)  OR the current message is a Feedback FLUSH the buffer
		if (actualSize == batchingLimit || (size_t)(this->mainBuffer_curr_ptr - this->buffers->payloads) >= sizeLimit || m->size == 0 || m->type == ChannelType::FBK)
			this->flush();

		MessageAllocator::releaseMessage(m);
	}
	return 0;
}

inline int uBuffer::flush(bool doNotReleaseBuffer) {
	// copy headers at the end of the main buffer
	memcpy(this->mainBuffer_curr_ptr, this->buffers->headers, actualSize * headerSize);
	// write the number of messages in the batch
	this->mainBuffer_curr_ptr += actualSize * headerSize;
	*(reinterpret_cast<int *>(this->mainBuffer_curr_ptr)) = htonl(this->actualSize);
	this->mainBuffer_curr_ptr += sizeof(int);

#ifdef DFF_ACK
	if (counter == 0 && parent->waitAckFrom(counter) == -1) {
		error("Errore waiting ack from socket inside the callback\n");
		return -1;
	}
#endif

	if (handle.isend(this->buffers->payloads, this->mainBuffer_curr_ptr - this->buffers->payloads, this->buffers->r) < 0) {
		error("flushing payload buffers");
		return -1;
	}

#ifdef DFF_ACK
	counter -= 1;
#endif
	// reset the parameters
	this->actualSize = 0;
	if (!doNotReleaseBuffer) {
		this->mainBuffer_curr_ptr = nullptr;
		uBufferAllocator::dealloc(this->buffers);
		this->buffers = nullptr;
	} else
		this->mainBuffer_curr_ptr = this->buffers->payloads;

	return 0;
}

} // namespace ff
#endif