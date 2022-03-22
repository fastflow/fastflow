#include "ff_network.hpp"
#include <sys/uio.h>
using namespace ff;

class ff_batchBuffer {
    std::function<bool(struct iovec*, int)> callback;
    int batchSize;
    struct iovec iov[1024];
    std::vector<std::pair<size_t*, message_t*>> toCleanup;
public:
    int size = 0;
    ff_batchBuffer() {}
    ff_batchBuffer(int _size, std::function<bool(struct iovec*, int)> cbk) : callback(cbk), batchSize(_size) {
        if (_size*4+1 > 1024){
            error("Size too big!\n");
            abort();
        }
        iov[0].iov_base = &(this->size);
        iov[0].iov_len = sizeof(int);
    }

    void push(message_t* m){
        m->sender = htonl(m->sender);
        m->chid = htonl(m->chid);
        size_t* sz = new size_t(htobe64(m->data.getLen()));


        int indexBase = size * 4;
        iov[indexBase+1].iov_base = &m->sender;
        iov[indexBase+1].iov_len = sizeof(int);
        iov[indexBase+2].iov_base = &m->chid;
        iov[indexBase+2].iov_len = sizeof(int);
        iov[indexBase+3].iov_base = sz;
        iov[indexBase+3].iov_len = sizeof(size_t);
        iov[indexBase+4].iov_base = m->data.getPtr();
        iov[indexBase+4].iov_len = m->data.getLen();

        toCleanup.emplace_back(sz, m);

        if (++size == batchSize)
            this->flush();
            
    }

    void sendEOS(){
        push(new message_t(0,0));
        flush();
    }

    void flush(){
        if (size == 0) return;

        int size_ = size;
        size = htonl(size);
        
        if (!callback(iov, size_*4+1))
            error("Callback of the batchbuffer got something wrong!\n");

         while (!toCleanup.empty()){
            delete toCleanup.back().first;
            delete toCleanup.back().second;
            toCleanup.pop_back();
        }
    

        size = 0;
    }

};