/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*
 * computes the n-th Fibonacci sequence number 
 */

#include <stdlib.h>
#include <iostream>
#include <stdint.h>
#include <ff/utils.hpp>

inline uint64_t fib(uint64_t n) {
    if (n<2) return n;    
    return fib(n-1)+fib(n-2);
}


int main(int argc, char * argv[]) {
    if (argc!=2) {
	std::cerr << "use: " << argv[0] << " n\n";
	return -1;
    }

    int n = atoi(argv[1]);
    ff::ffTime(ff::START_TIME);
    uint64_t  f = fib(n);
    ff::ffTime(ff::STOP_TIME);
    std::cout << "fib(" << n << ")= " << f << "\n";
    printf("Time: %g (ms)\n", ff::ffTime(ff::GET_TIME));
    return 0;
}
