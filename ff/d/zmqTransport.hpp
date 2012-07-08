/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
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

#ifndef _FF_ZMQTRANSPORT_HPP_
#define _FF_ZMQTRANSPORT_HPP_

// TODO: better management of error (and exception) conditions.
//
//

// NOTE: from 0mq manuals (http://zguide.zeromq.org/)
//   
//   You MUST NOT share ØMQ sockets between threads. 
//   ØMQ sockets are not threadsafe. Technically it's possible to do 
//   this, but it demands semaphores, locks, or mutexes. This will make 
//   your application slow and fragile. The only place where it's 
//   remotely sane to share sockets between threads are in language 
//   bindings that need to do magic like garbage collection on sockets.
//

#include <assert.h>
#include <deque>
#include <algorithm>

#include <zmq.hpp>

namespace ff {


class zmqTransportMsg_t: public zmq::message_t {    
public:
    // callback definition 
    typedef void (msg_callback_t) (void *, void *);

    enum {HEADER_LENGHT=4}; // n. of bytes

public:
    inline zmqTransportMsg_t() {}

    inline zmqTransportMsg_t(const void * data, size_t size, msg_callback_t cb=0,void * arg=0):
        zmq::message_t(const_cast<void*>(data),size,cb,arg) {}


    inline void init(const void * data, size_t size, msg_callback_t cb=0,void * arg=0) {
        zmq::message_t::rebuild(const_cast<void*>(data),size,cb,arg);
    }

    inline void copy(zmq::message_t &msg) {
        zmq::message_t::copy(&msg);
    }

    inline void * getData() {
        return zmq::message_t::data();
    }

    inline size_t size() {
        return zmq::message_t::size();
    }
};

class zmqTransport {
private:
    enum {NUM_IO_THREADS=1};

protected:

    // closes all connections
    inline int closeConnections() {
        //
        // WARNING: Instead of setting a linger period of some minutes, 
        // it would be better to implement a shoutdown protocol and to set 
        // the linger period to 0.
        //
        int zero = 1000000; //milliseconds 
        for(unsigned i=0;i<Socks.size();++i) {
            if (Socks[i]) { 
                Socks[i]->setsockopt(ZMQ_LINGER, &zero, sizeof(zero));
                Socks[i]->close();
                delete Socks[i];
                Socks[i]=NULL;
            }
        }
        return 0;
    }

public:
    typedef zmq::socket_t endpoint_t;
    typedef zmqTransportMsg_t msg_t;

    zmqTransport(const int procId) :
        procId(procId),context(NULL) {
    };

    ~zmqTransport() { closeTransport();  }

    int initTransport() {
        if (context) return -1;
        context = new zmq::context_t(NUM_IO_THREADS);        
        if (!context) return -1;
        if (Socks.size()>0) closeConnections();
        return 0;
    }
    
    int closeTransport() {
        closeConnections();
        if (context) {delete context; context=NULL;}        
        return 0;
    }

    
    endpoint_t * newEndPoint(const bool P) {
        endpoint_t * s = new endpoint_t(*context, (P ? ZMQ_DEALER : ZMQ_ROUTER));
        if (!s) return NULL;
        Socks.push_back(s);
        return s;
    }
    
    int deleteEndPoint(endpoint_t *s) {
        if (s) {
            std::deque<endpoint_t*>::iterator it = std::find(Socks.begin(), Socks.end(), s);
            if (it != Socks.end()) {
                Socks.erase(it);

                // WARNING: see closeConnections.
                //
                int zero = 1000000; //milliseconds 
                s->setsockopt(ZMQ_LINGER, &zero, sizeof(zero));
                s->close();
                delete s;
                return 0;
            }
        }
        return -1;
    }

    const int getProcId() const { return procId;}

protected:
    const int                procId;
    zmq::context_t *         context;
    std::deque<endpoint_t *> Socks;    // all active end-points
};

} // namespace
#endif /* _ZMQTRANSPORT_HPP_INCLUDE_ */
