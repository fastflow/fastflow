/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU General Public License version 2 as published by
 *  the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License for
 *  more details.
 *
 *  You should have received a copy of the GNU General Public License along
 *  with this program; if not, write to the Free Software Foundation, Inc., 59
 *  Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this file
 *  does not by itself cause the resulting executable to be covered by the GNU
 *  General Public License.  This exception does not however invalidate any
 *  other reasons why the executable file might be covered by the GNU General
 *  Public License.
 *
 * **************************************************************************/

#ifndef FF_STATIC_ALLOCATOR_HPP
#define FF_STATIC_ALLOCATOR_HPP

#include <sys/mman.h>
#include <cstdlib>
#include <cstdio>
#include <ff/config.hpp>
#include <atomic>
#include <new>

/* Author:
 *  Massimo Torquati
 *
 */


namespace ff {

class StaticAllocator {
public:
	StaticAllocator(const size_t nslot, const size_t slotsize):
		nslot(nslot),slotsize(slotsize), cnt(0), segment(nullptr) {

		void* result = 0;
        result = mmap(NULL, nslot*slotsize, PROT_READ|PROT_WRITE, MAP_PRIVATE|MAP_ANONYMOUS, -1, 0);
        if (result == MAP_FAILED) abort();
        segment = (char*)result;
        printf("allocated %ld bytes\n", nslot*slotsize);
	}

	~StaticAllocator() {
		if (segment) munmap(segment, nslot*slotsize);
		segment=nullptr;
	}

	template<typename T>
	void alloc(T*& p) {
		size_t m  = cnt.fetch_add(1,std::memory_order_relaxed) % nslot;
		char  *mp = segment + m*slotsize;
		new (mp) T();
		p = reinterpret_cast<T*>(mp);
	}

	template<typename T, typename S>
	void realloc(T* in, S*& out) {
        dealloc(in);
        char  *mp = reinterpret_cast<char*>(in);
		new (mp) S();
		out = reinterpret_cast<S*>(mp);
	}
    
protected:

	template<typename T>
	void dealloc(T* p) {
		p->~T();
	}
    
private:
	const size_t nslot;
	const size_t slotsize;
	std::atomic_ullong cnt;
	char *segment;
};

};


#endif /* FF_STATIC_ALLOCATOR_HPP */
