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

/**
  Fibonacci: computes the n-th number of the fibonacci sequence using the ff_DAC pattern
  */

#include <iostream>
#include <functional>
#include <vector>
#include <ff/dc.hpp>
using namespace ff;
using namespace std;

/*
 * Operand and Result are just integers
 */
using Problem = long;
using Result  = long;

int main(int argc, char *argv[]) {
    long start = 20;
    long nwork = 4;
    if (argc>1) {
        if(argc<3){
            fprintf(stderr,"Usage: %s <N> <pardegree>\n",argv[0]);
            return -1;
        }
        start=atoi(argv[1]);
        nwork=atoi(argv[2]);
    }
    
	long res;
	//lambda version 
	ff_DC<long, long> dac(
                          [](const Problem &op,std::vector<Problem> &subops){
                              subops.push_back(op-1);
                              subops.push_back(op-2);
                          },
                          [](vector<Result>& res, Result &ret){ ret=res[0]+res[1]; },
                          [](const Problem &, Result &res)  { res=1; },
                          [](const Problem &op){ return (op<=2); },
                          Problem(start),  res,  nwork
                          );
	ffTime(START_TIME);
	//compute
	if (dac.run_and_wait_end()<0) { 
        error("running dac");
        return -1;
    }
	ffTime(STOP_TIME);
	printf("Result: %ld\n",res);
	printf("Time (usecs): %g\n",ffTime(GET_TIME));
    return 0;
}
