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

// MarcoA
// Part of FastFlow tests
// Aug 3, 1024
// All parfors inc element-wise the array A.

/**
 * \file parfor_basic.cpp
 * \ingroup applications
 * \brief Some basic usage examples of the parfor pattern
 *
 * @include parfor_basic.cpp
 */

#include <iostream>
#include <iomanip>
#include <sstream>
#include <ff/parallel_for.hpp>
 
const long N=10;

static void print(long V[], long size) {
  for (long i=0;i<size;++i) std::cout << std::setw(2) << V[i] << " ";
  std::cout << std::endl;
}

static void compute_mask(long V[], long size, std::string m) {
  std::cout << "Compute mask (X anonymous worker, - not computed)" << std::endl;
  for (long i=0;i<size;++i) 
    if (V[i]!=i) std::cout <<  std::setw(2) << m << " ";
    else std::cout << " -" << " ";
  std::cout << std::endl;
}

static void compute_mask_i(long V[], long size, long mapping[]) {
  std::cout << "Compute mask (worker ID, - not computed)" << std::endl;
  for (long i=0;i<size;++i) 
      if (V[i]!=i) std::cout <<  std::setw(2) << std::to_string(mapping[i]) << " ";
    else std::cout << " -" << " ";
  std::cout << std::endl;
}

static void reset(long V[], long size) {
  for (long i=0;i<size;++i) V[i] =i;
}



int main() {
  long nworkers = 2;
  long A[N];
  for (long i=0;i<N;++i) A[i]=i;


  /* 1) Parallel for region (basic) 

     void parallel_for (long first, long last, const Function &f, const long nw=FF_AUTO)

    Compute the body on [first,last) elements
    Dynamic scheduling. grain = N/nw
  */

  ff::ParallelFor pf(nworkers, false);
  pf.parallel_for(0L,N,[&A](const long i) {
      A[i]+=1;
    });
  /* OpenMP version 
   * #pragma omp parallel for num_threads(nworkers)
   * for(long i=0;i<N;++i) {
   * A[i]+=1;
   * };
   */
  
  std::cout << "====================================================" << std::endl;
  std::cout << "1) Basic" << std::endl;
  print(A,N);
  compute_mask(A,N," X");
  reset(A,N);

  /* 2) Parallel for region (step)
     
     parallel_for (long first, long last, long step, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Dynamic scheduling, grain = (last-first)/(step*nw)

  */
  
  long step = 2;
  pf.parallel_for(0L,N,step,[&A](const long i) {
      A[i]+=1;
    }, nworkers);

  std::cout << "====================================================" << std::endl;
  std::cout << "2) Step" << step << std::endl;
  print(A,N);
  compute_mask(A,N," X");
  reset(A,N);

  /* 3) Parallel for region (step, grain) 
     
     parallel_for (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Dynamic scheduling, grain defined by user
  */
  long grain = 3;
  long mapping_on_threads[N];
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;

  pf.parallel_for(0L,N,step,grain,[&A,&mapping_on_threads](const long i) {
      A[i]+=1;
      // Internal, OS-dependent thread ID
      mapping_on_threads[i] = ff_getThreadID(); 
    });

  std::cout << "====================================================" << std::endl;
  std::cout << "3) step << " << step << " grain=" << grain << std::endl;
  print(A,N);
  compute_mask(A,N," X");
  for (long i=0;i<N;++i) {
    std::cout << "A["<< i << "] on Thread ";
    if (mapping_on_threads[i]==-1) std::cout << "NA";
    else std::cout << mapping_on_threads[i];
    std::cout << std::endl;
  }
  reset(A,N);

  /* 4) Parallel for region with threadID (step, grain, thid)
     
     parallel_for_thid (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Dynamic scheduling, chunk defined by user, body can access to Worker ID
  */
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;

  pf.parallel_for_thid(0L,N,step,grain,[&A,&mapping_on_threads](const long i, const int thid) {
      A[i]+=1;
      mapping_on_threads[i] = thid;
    });

  std::cout << "====================================================" << std::endl;
  std::cout << "4) step= " << step << "grain = " << grain << 
    "with ThreadIDs" << std::endl;
  print(A,N);
  compute_mask_i(A,N,mapping_on_threads);
  reset(A,N);

   /* Parallel for region with indexes ranges (step, grain, thid, idx)
     
     parallel_for_idx (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Dynamic scheduling, chunk defined by user, body can access to worker number

  */

  std::cout << "====================================================" << std::endl;
  std::cout << "5) Partition range indexes (IDX)" << std::endl; 
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;
  pf.parallel_for_idx(0L,N,3,2,[&A,&mapping_on_threads](const long start, const long end, const long thid) {
      usleep(random()&1111111);
      std::cerr << start << " - " << end << "\n";
      for (long j=start; j<end; j+=3) {
	A[j]+=1;
	mapping_on_threads[j] = thid;
      }
    },nworkers);

  print(A,N);
  compute_mask_i(A,N,mapping_on_threads);
  reset(A,N);

  /* Parallel for region static (step, grain)
     
     parallel_for_static (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Static scheduling, grain defined by user or with maximal partitions

  */
  std::cout << "====================================================" << std::endl;
  step = 1;
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;
  pf.parallel_for_static(0L,N,step,0,[&A,&mapping_on_threads](const long i) {

      A[i]+=1;
      // Internal, OS-dependent thread ID
      mapping_on_threads[i] = ff_getThreadID(); 
    });

  std::cout << "6) Static with maximal partitions" << std::endl; 
  print(A,N);
  compute_mask(A,N," X");
  for (long i=0;i<N;++i) {
    std::cout << "A["<< i << "] on Thread ";
    if (mapping_on_threads[i]==-1) std::cout << "NA";
    else std::cout << mapping_on_threads[i];
    std::cout << std::endl;
  }
  reset(A,N);

  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;
  pf.parallel_for_static(0L,N,step, grain, [&A,&mapping_on_threads](const long i) {
      A[i]+=1;
      // Internal, OS-dependent thread ID
      mapping_on_threads[i] = ff_getThreadID(); 
    },nworkers);
  
  std::cout << "====================================================" << std::endl;
  std::cout << "7) Static with fixed partition size = " << grain << std::endl; 
  print(A,N);
  compute_mask(A,N," X");
  for (long i=0;i<N;++i) {
    std::cout << "A["<< i << "] on Thread ";
    if (mapping_on_threads[i]==-1) std::cout << "NA";
    else std::cout << mapping_on_threads[i];
    std::cout << std::endl;
  }
  reset(A,N);

}
	
  
