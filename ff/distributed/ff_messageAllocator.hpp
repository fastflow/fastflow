#ifndef MESALL_HPP
#define MESALL_HPP
#include "ff_ddefines.hpp"
#include "ff/mpmc/MPMCqueues.hpp"

namespace ff {
#ifndef NO_MESSAGE_ALLOCATOR
class MessageAllocator {
    inline static MPMC_Ptr_Queue messageAllocatorQueue;

public:
    static void init(size_t s){
        messageAllocatorQueue.init(s);
        for(size_t i = 0; i < s; i++)
            messageAllocatorQueue.push(new message2_t);
    }

    static void finalize(){
        message2_t* tmp;
        while(messageAllocatorQueue.pop((void**)&tmp))
            delete tmp;
    }

    static inline message2_t* allocateMessage(){
        message2_t* out;
        if (messageAllocatorQueue.pop((void**)&out)) return out;
        return new message2_t;
    }

    static inline void releaseMessage(message2_t* m){
        m->cleanContent();
        if (!messageAllocatorQueue.push(m))
            delete m;
    }

    inline static message2_t* make_logical_EOS(int src, int dest = -1){
        message2_t* out = allocateMessage();
        out->dest = dest;
        out->src = src;
        return out;
    }

    inline static message2_t* make_flush(int src){
        message2_t* out = allocateMessage();
        out->dest = -2; // tag for flushing command
        out->src = src;
        return out;
    }

};
#else
struct MessageAllocator {
    static void init(size_t){}
    static void finalize(){}
    static inline message2_t* allocateMessage(){ return new message2_t; }
    static inline void releaseMessage(message2_t* m){ delete m; }
    inline static message2_t* make_logical_EOS(int src, int dest = -1){
        message2_t* out = new message2_t;
        out->dest = dest;
        out->src = src;
        return out;
    }

    inline static message2_t* make_flush(int src){
        message2_t* out = new message2_t;
        out->dest = -2; // tag for flushing command
        out->src = src;
        return out;
    }

};
#endif

}
#endif