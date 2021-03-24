#ifndef FF_NETWORK
#define FF_NETWORK

#include <sstream>
#include <iostream>
#include <exception>
#include <string>
#include <sys/socket.h>
#include <sys/un.h>

#include <sys/types.h>
#include <sys/uio.h>
#include <arpa/inet.h>

//#define LOCAL
#define REMOTE

#ifdef __APPLE__
    #include <libkern/OSByteOrder.h>
    #define htobe16(x) OSSwapHostToBigInt16(x)
    #define htole16(x) OSSwapHostToLittleInt16(x)
    #define be16toh(x) OSSwapBigToHostInt16(x)
    #define le16toh(x) OSSwapLittleToHostInt16(x)

    #define htobe32(x) OSSwapHostToBigInt32(x)
    #define htole32(x) OSSwapHostToLittleInt32(x)
    #define be32toh(x) OSSwapBigToHostInt32(x)
    #define le32toh(x) OSSwapLittleToHostInt32(x)

    #define htobe64(x) OSSwapHostToBigInt64(x)
    #define htole64(x) OSSwapHostToLittleInt64(x)
    #define be64toh(x) OSSwapBigToHostInt64(x)
    #define le64toh(x) OSSwapLittleToHostInt64(x)
#endif

class dataBuffer: public std::stringbuf {
public:	
    dataBuffer()
        : std::stringbuf(std::ios::in | std::ios::out | std::ios::binary) {
	}
    dataBuffer(char p[], size_t len, bool cleanup=false)
        : std::stringbuf(std::ios::in | std::ios::out | std::ios::binary),
		  len(len),cleanup(cleanup) {
        setg(p, p, p + len);
    }
	~dataBuffer() {
		if (cleanup) delete [] getPtr();
	}
	size_t getLen() const {
		if (len>=0) return len;
		return str().length();
	}
	char* getPtr() const {
		return eback();
	}

	void doNotCleanup(){
		cleanup = false;
	}

protected:	
	ssize_t len=-1;
	bool cleanup = false;
};

struct message_t {
	message_t(){}
	message_t(char *rd, size_t size, bool cleanup=true) : data(rd,size,cleanup){}
	
	int           sender;
	int           chid;
	dataBuffer    data;
};


struct ff_endpoint {
	std::string address;
	int port;
};


struct FF_Exception: public std::runtime_error {FF_Exception(const char* err) throw() : std::runtime_error(err) {}};

ssize_t readn(int fd, char *ptr, size_t n) {  
    size_t   nleft = n;
    ssize_t  nread;

    while (nleft > 0) {
        if((nread = read(fd, ptr, nleft)) < 0) {
            if (nleft == n) return -1; /* error, return -1 */
            else break; /* error, return amount read so far */
        } else if (nread == 0) break; /* EOF */
        nleft -= nread;
        ptr += nread;
    }
    return(n - nleft); /* return >= 0 */
}

ssize_t readvn(int fd, struct iovec *v, int count){
    ssize_t rread;
    for (int cur = 0;;) {
        rread = readv(fd, v+cur, count-cur);
        if (rread <= 0) return rread; // error or closed connection
        while (cur < count && rread >= (ssize_t)v[cur].iov_len)
            rread -= v[cur++].iov_len;
        if (cur == count) return 1; // success!!
        v[cur].iov_base = (char *)v[cur].iov_base + rread;
        v[cur].iov_len -= rread;
    }
}

ssize_t writen(int fd, const char *ptr, size_t n) {  
    size_t   nleft = n;
    ssize_t  nwritten;
    
    while (nleft > 0) {
        if((nwritten = write(fd, ptr, nleft)) < 0) {
            if (nleft == n) return -1; /* error, return -1 */
            else break; /* error, return amount written so far */
        } else if (nwritten == 0) break; 
        nleft -= nwritten;
        ptr   += nwritten;
    }
    return(n - nleft); /* return >= 0 */
}

ssize_t writevn(int fd, struct iovec *v, int count){
    ssize_t written;
    for (int cur = 0;;) {
        written = writev(fd, v+cur, count-cur);
        if (written < 0) return -1;
        while (cur < count && written >= (ssize_t)v[cur].iov_len)
            written -= v[cur++].iov_len;
        if (cur == count) return 1; // success!!
        v[cur].iov_base = (char *)v[cur].iov_base + written;
        v[cur].iov_len -= written;
    }
}

#endif
