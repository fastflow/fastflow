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
/* Author: Massimo
 * Date  : July 2014
 *         
 */
/*
 * This test tests the ff_Map pattern (aka ParallelForReudce) in a pipeline computation.
 * The structure is:
 *
 *    pipeline( farm(mapWorker,2) , mapStage );
 *
 */

#include <vector>
#include <ff/config.hpp>
#if !defined(HAS_CXX11_VARIADIC_TEMPLATES)
#define HAS_CXX11_VARIADIC_TEMPLATES 1
#endif
#include <ff/parallel_for.hpp>
#include <ff/ff.hpp>
#include <ff/map.hpp>


using namespace ff;
const long MYSIZE = 100;

typedef  std::vector<long> ff_task_t;

#ifdef min
#undef min //MD workaround to avoid clashing with min macro in minwindef.h
#endif

struct mapWorker : ff_Map<ff_task_t> {
    ff_task_t *svc(ff_task_t *) {
        ff_task_t *A = new ff_task_t(MYSIZE);
       
        // this is the parallel_for provided by the ff_Map class
        parallel_for(0,A->size(),[&A](const long i) { 
                A->operator[](i)=i;
		}, std::min(3, (int)ff_realNumCores()));
        ff_send_out(A);
        return EOS;
    }
};


struct mapStage: ff_Map<ff_task_t> {
    ff_task_t *svc(ff_task_t *inA) {
        ff_task_t &A = *inA;
        // this is the parallel_for provided by the ff_Map class
        parallel_for(0,A.size(),[&A](const long i) { 
                A[i] += i;
            },2);
        
        printf("mapStage received:\n");
        for(size_t i=0;i<A.size();++i)
            printf("%ld ", A[i]);
        printf("\n");
        delete inA;
        return GO_ON;
    }    
};


int main() {
    
    // farm having map workers
    ff_Farm<ff_task_t, ff_task_t> farm( []() {
            std::vector<std::unique_ptr<ff_node> > W;	
            for(size_t i=0;i<2;++i) W.push_back(make_unique<mapWorker>());
            return W;
        }() );
    
    mapStage stage;
    ff_Pipe<> pipe(farm,stage);
    if (pipe.run_and_wait_end()<0)
        error("running pipe");   
    return 0;	
}
