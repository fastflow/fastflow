#ifndef SERBUFFER_HPP
#define SERBUFFER_HPP
#include <sstream>
#include <streambuf>

class serBuffer : public std::streambuf {
public:
    serBuffer(size_t initialSize = 2)
        : m_buffer(nullptr), m_capacity(0) {
        resizeBuffer(initialSize);
    }

    ~serBuffer() {
        if (m_buffer) free(m_buffer);
    }

    // Retrieve the current buffer
    char* buffer() const {
        return m_buffer;
    }

    // Get the size of the data written so far
    size_t size() const {
        return pptr() - pbase();
    }

    // Change the size of the buffer
    void resizeBuffer(size_t newCapacity) {
        if (newCapacity <= m_capacity) {
            return; // No need to resize if the new capacity is smaller
        }

        char* newBuffer = static_cast<char*>(realloc(m_buffer, newCapacity));
        if (!newBuffer) {
            throw std::bad_alloc();
        }

        // Adjust pointers
        std::ptrdiff_t currentSize = pptr() - pbase();
        m_buffer = newBuffer;
        m_capacity = newCapacity;
        setp(m_buffer, m_buffer + m_capacity);
        pbump(static_cast<int>(currentSize));
    }

    std::pair<char*, size_t> getBufferAndReset(){
        auto out = std::make_pair(m_buffer, size());
        m_buffer = static_cast<char*>(malloc(m_capacity));
        setp(m_buffer, m_buffer + m_capacity);
        return out;
    }

protected:
    int overflow(int ch) override {
        if (ch != EOF) {
            resizeBuffer(m_capacity * 2); // Double the buffer capacity
            *pptr() = static_cast<char>(ch);
            pbump(1);
            return ch;
        }
        return EOF;
    }

private:
    char* m_buffer;     // Pointer to the buffer
    size_t m_capacity;  // Total allocated capacity
};

struct deserBuff: public std::stringbuf {
    deserBuff() : std::stringbuf( std::ios::in | std::ios::binary) {}

    void setBuffer(char p[], size_t len){
        setg(p, p, p+len);
    }
};

#endif