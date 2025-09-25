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
 *  luca Ferrucci
 */

#ifndef FF_FAAS_BUFFER_HPP
#define FF_FAAS_BUFFER_HPP

#include <streambuf>
#include <cstring>  
#include <functional>

namespace ff {

    class faasBuffer : public std::streambuf {
    public:
        faasBuffer(size_t initialSize = 0)
            : freetaskF(nullptr),m_buffer(nullptr),m_capacity(0),m_tempBuffer(nullptr),m_tempLen(0), m_cleanup(false)  {
            resizeBuffer(initialSize);
        }

        ~faasBuffer() {
            cleanTempBuffer();  
            if(m_buffer) 
                delete[] m_buffer;
        }

        // Dimensione dei dati scritti finora
        size_t size() const {
            return static_cast<size_t>(epptr() - pbase());   
        }

        // Puntatore al buffer corrente
        char* getBuffer() const {
            if (m_tempBuffer) 
                return m_tempBuffer;
            return m_buffer;
        }

        void reuseBuffer(size_t newCapacity) {            
            resizeBuffer(newCapacity);
            cleanTempBuffer();  
        }

        void setCleanup(bool cleanup) {
            m_cleanup = cleanup;
        }

        void setBuffer(char* p, size_t len, bool cleanupBuffer = true) {
            cleanTempBuffer();  
            m_tempBuffer = p;
            m_tempLen = len;
            m_cleanup = cleanupBuffer;
            setp(m_tempBuffer, m_tempBuffer + m_tempLen);
            setg(m_tempBuffer, m_tempBuffer, m_tempBuffer + m_tempLen);
        }

    	std::function<void(void*)> freetaskF;

    protected:
        
        int overflow(int ch) override {
            if (ch != EOF) {
                if (!m_tempBuffer) {
                    size_t currentPos = pptr() - m_buffer;                    
                    if (currentPos >= m_capacity)         
                        resizeBuffer(m_capacity * 2);                    
                    setp(m_buffer, m_buffer + m_capacity);
                    pbump(static_cast<int>(currentPos));
                    *pptr() = static_cast<char>(ch);
                    pbump(1);
                    return ch;
                } else 
                    return EOF;                
            }
            return EOF;
        }

    private:
        char* m_buffer;    
        size_t m_capacity; 
        char* m_tempBuffer; 
        size_t m_tempLen;   
        bool m_cleanup;     

        void cleanTempBuffer() {
            if (m_tempBuffer) {
                if (m_cleanup) {
                    if (freetaskF) 
                        freetaskF(m_tempBuffer);
                    else 
                        delete[] m_tempBuffer;
                }
                m_tempBuffer = nullptr;
                m_tempLen = 0;
                m_cleanup = false;
            }
        }

        void resizeBuffer(size_t newCapacity) {
            if(newCapacity<=m_capacity) {
                setp(m_buffer, m_buffer + m_capacity);
                return;
            }
            if (m_buffer) 
                delete[] m_buffer;
            m_buffer = new char[newCapacity];
            m_capacity = newCapacity;
            setp(m_buffer, m_buffer + m_capacity);
            setg(m_buffer,m_buffer, m_buffer + m_capacity);
        }
    };

} // namespace ff

#endif // FF_FAAS_BUFFER
