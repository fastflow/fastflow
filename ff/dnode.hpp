/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_DNODE_HPP_
#define _FF_DNODE_HPP_
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

#include <sys/uio.h>
#include <stdint.h>
#include <cstdlib>
#include <ff/d/zmqTransport.hpp>
#include <ff/d/zmqImpl.hpp>
#include <ff/node.hpp>
#include <ff/svector.hpp>


namespace ff {

// the header is msg_t::HEADER_LENGHT bytes
static const char FF_DEOS[]="EOS";

// callback definition 
typedef void (*dnode_cbk_t) (void *,void*);

template <typename CommImpl>
class ff_dnode: public ff_node {
public:
    typedef typename CommImpl::TransportImpl::msg_t msg_t;
protected:    
    // this is called once as soon as the message just sent 
    // is no longer in use by the run-time
    static dnode_cbk_t cb;
    
    static inline void freeMsg(void * data, void * arg) {
        if (cb) cb(data,arg);
    }
    static inline void freeHdr(void * data, void *) {
        delete static_cast<uint32_t*>(data);
    }

    inline bool isEos(const char data[msg_t::HEADER_LENGHT]) const {
        return !(data[0]-'E' + data[1]-'O' + data[2]-'S' + data[3]-'\0');
    }
    
    // default constructor
    ff_dnode():ff_node(),skipdnode(true) {}
    
    virtual ~ff_dnode() {         
        com.close();
        delete com.getDescriptor();
    }
    
    virtual inline bool push(void * ptr) { 
        if (skipdnode || !P) return ff_node::push(ptr);

        // gets the peers involved in one single communication
        const int peers=com.putToPerform(); 

        if (ptr == (void*)FF_EOS) {
            for(int i=0;i<com.getDescriptor()->getPeers();++i) {
                msg_t msg; 
                msg.init(FF_DEOS,msg_t::HEADER_LENGHT);
                if (!com.put(msg,i)) return false;
            }
            return true;
        }

        svector<iovec> v[peers];
        for(int i=0;i<peers;++i) {
            callbackArg.resize(0);
            prepare(v[i], ptr, i);

            uint32_t *len = new uint32_t(v[i].size());
            msg_t hdr(len, msg_t::HEADER_LENGHT, freeHdr);
            com.putmore(hdr);
            callbackArg.resize(*len);
            for(size_t j=0;j<v[i].size()-1;++j) {
                msg_t msg(v[i][j].iov_base, v[i][j].iov_len,freeMsg,callbackArg[j]); 
                com.putmore(msg);
            }
            msg_t msg(v[i][*len-1].iov_base, v[i][*len-1].iov_len,freeMsg,callbackArg[*len-1]);
            if (!com.put(msg)) return false;
        }
        return true;
    }
    
    virtual inline bool pop(void ** ptr) { 
        if (skipdnode || P) return ff_node::pop(ptr);

        // gets the peers involved in one single communication
        const int sendingPeers=com.getToWait();

        svector<msg_t*>* v[sendingPeers];
        for(int i=0;i<sendingPeers;++i) {
            msg_t hdr;
            int sender=-1;
            if (!com.gethdr(hdr, sender)) {
                error("dnode:pop: ERROR: receiving header from peer");
                return false;
            }
            
            if (isEos(static_cast<char *>(hdr.getData()))) {
                if (++neos==com.getDescriptor()->getPeers()) {
                    com.done();
                    *ptr = (void*)FF_EOS;
                    neos=0;
                    return true;
                }
                if (sendingPeers==1) i=-1; // reset for index
                continue;
            }
            uint32_t len = *static_cast<uint32_t*>(hdr.getData());
            register int ventry   = (sendingPeers==1)?0:sender;
            prepare(v[ventry], len, sender);
            assert(v[ventry]->size() == len);

            for(size_t j=0;j<len;++j)
                if (!com.get(*(v[ventry]->operator[](j)))) {
                    error("dnode:pop: ERROR: receiving data from peer");
                    return false;
                }
        }
        com.done();
        
        unmarshalling(v, sendingPeers, *ptr);
        return true;
    } 
    
public:

    int init(const std::string& name, const std::string& address,
             const int peers, typename CommImpl::TransportImpl* const transp, 
             const bool p, const int nodeId=-1, dnode_cbk_t cbk=0) {
        if (!p && cbk) {
            error("dnode:init: WARNING: callback does not make sense for consumer end-point, ignoring it...\n");
            cbk=NULL;
        }
        skipdnode=false;
        P=p; 
        neos=0;
        ff_dnode::cb=cbk;                               
        if (P) ff_node::create_output_buffer(1,true);            
        else ff_node::create_input_buffer(1);
        com.setDescriptor(new typename CommImpl::descriptor(name,peers,transp,P));
        return com.init(address,nodeId);
    }
        
    // serialization/deserialization methods

    // Used by the PRODUCER
    // Used to prepare (non contiguous) output messages
    virtual void prepare(svector<iovec>& v, void* ptr, const int sender=-1) {
        struct iovec iov={ptr,sizeof(void*)};
        v.push_back(iov);
        setCallbackArg(NULL);
    }

    // COMMENT: 
    //  When using ZeroMQ (from zguide.zeromq.org):
    //   "There is no way to do zero-copy on receive: ØMQ delivers you a 
    //    buffer that you can store as long as you wish but it will not 
    //    write data directly into application buffers."
    //
    //
    
    // Used by the CONSUMER
    // Used to give to the run-time a pool of messages on which
    // input data can be received
    // @param len is the number of input messages expected
    // @param sender is the message sender
    virtual void prepare(svector<msg_t*>*& v, size_t len, const int sender=-1) {
        svector<msg_t*> * v2 = new svector<msg_t*>(len);
        assert(v2);
        for(size_t i=0;i<len;++i) {
            msg_t * m = new msg_t;
            assert(m);
            v2->push_back(m);
        }
        v = v2;
    }
    
    // Used by the CONSUMER
    //
    virtual void unmarshalling(svector<msg_t*>* const v[], const int vlen, void *& task) {
        assert(vlen==1 && v[0]->size()==1); 
        task = v[0]->operator[](0)->getData();
        delete v[0];
    }

    // this methed can be used to pass an additional parameter (the 2nd one) 
    // to the callback function. Typically it is called in the prepare method
    void setCallbackArg(void* arg) { callbackArg.push_back(arg);}
    
    int  run(bool=false) { return  ff_node::run(); }    
    int  wait() { return ff_node::wait(); }    
    void skipfirstpop(bool sk)   { ff_node::skipfirstpop(sk); }

protected:
    bool     skipdnode;
    bool     P;   
    int      neos;
    svector<void*> callbackArg;
    CommImpl com;
};
template <typename CommImpl>
dnode_cbk_t ff_dnode<CommImpl>::cb=0;
} // namespace ff

#endif /* _FF_NODE_HPP_ */
