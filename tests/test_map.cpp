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
 * This test tests the ff_Map pattern (aka ParallelForReudce) in a pipeline computation.
 * The structure is:
 *
 *    pipeline( farm(mapWorker,2) , mapStage );
 *
 */

#include <ff/parallel_for.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/map.hpp>

using namespace ff;


const long SIZE = 100;

class mapWorker: public ff_Map<> {
public:
    void *svc(void*) {
	std::vector<long> *A = new std::vector<long>(SIZE);
	
	// this is the parallel_for provided by the ff_Map class
	parallel_for(0,A->size(),[&A](const long i) { 
		A->operator[](i)=i;
	    });
	ff_send_out(A);
	return NULL;
    }
};

class mapStage: public ff_Map<> {
public:
    void *svc(void *task) {
	std::vector<long> *A = reinterpret_cast<std::vector<long>*>(task);

	// this is the parallel_for provided by the ff_Map class
	parallel_for(0,A->size(),[&A](const long i) { 
		A->operator[](i) += i;
	    });

	printf("mapStage received:\n");
	for(size_t i=0;i<A->size();++i)
	    printf("%ld ", A->operator[](i));
	printf("\n");
	
	return GO_ON;
    }    
};


int main() {
    ff_pipeline pipe;
    std::vector<ff_node*> W;
    W.push_back(new mapWorker);
    W.push_back(new mapWorker);
    ff_farm<> farm(W);
    farm.cleanup_workers();
    pipe.add_stage(&farm);
    mapStage stage;
    pipe.add_stage(&stage);
    pipe.run_and_wait_end();

    return 0;
	
}
