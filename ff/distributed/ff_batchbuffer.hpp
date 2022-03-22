#include "ff_network.hpp"
#include <sys/uio.h>
using namespace ff;

class ff_batchBuffer {
    std::function<bool(struct iovec*, int)> callback;
    int batchSize;
    struct iovec iov[1024];
public:
    int size = 0;
    ff_batchBuffer() {}
    ff_batchBuffer(int _size, std::function<bool(struct iovec*, int)> cbk) : callback(cbk), batchSize(_size) {
        if (_size*2+1 > 1024){
            error("Size too big!\n");
            abort();
        }
        iov[0].iov_base = &(this->size);
        iov[0].iov_len = sizeof(int);
    }

    void push(message_t* m){
        
        ff::cout << "Pushing something of size: " << m->data.getLen();

        int sender_ = htonl(m->sender);
        int chid_ = htonl(m->chid);
        size_t size_ = htobe64(m->data.getLen());

        char* buffer = new char[sizeof(int)+sizeof(int)+sizeof(size_t)+1];
        memcpy(buffer, &sender_, sizeof(int));
        memcpy(buffer+sizeof(int), &chid_, sizeof(int));
        memcpy(buffer+sizeof(int)+sizeof(int), &size_, sizeof(size_t));

        /**reinterpret_cast<int*>(buffer) = htonl(m->sender);
        *reinterpret_cast<int*>(buffer+sizeof(int)) = htonl(m->chid);
        *reinterpret_cast<size_t*>(buffer+sizeof(int)+sizeof(int)) = htobe64(m->data.getLen());
        */

        int indexBase = size * 2;
        iov[indexBase+1].iov_base = buffer;
        iov[indexBase+1].iov_len = sizeof(int)+sizeof(int)+sizeof(size_t);
        iov[indexBase+2].iov_base = m->data.getPtr();
        iov[indexBase+2].iov_len = m->data.getLen();

        m->data.doNotCleanup();
        delete m;

        if (++size == batchSize)
            this->flush();
            
    }

    void sendEOS(){
        push(new message_t(0,0));
        flush(true);
    }

    void flush(bool last = false){
        if (size == 0) return;
        
        ff::cout << "Sending out a buffer os size: " << size << std::endl;
        if (!callback(iov, size*2+1-last))
            error("Callback of the batchbuffer got something wrong!\n");

        for(size_t i = 0; i < size; i++){
            
            iov[i*2+1].iov_len = 0;
            delete [] (char*)iov[i*2+1].iov_base;
            if (last && i == (size-1))  break;
            iov[i*2+2].iov_len = 0;
            delete [] (char*)iov[i*2+2].iov_base;
        }

        size = 0;
    }

};