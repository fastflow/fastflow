// MarcoA
// Part of FastFlow tests
// Aug 3, 1024

#include <iostream>
#include <iomanip>
#include <sstream>
#ifdef FF
#include <ff/parallel_for.hpp>
#else
# include <omp.h>
#endif
 
#define N 10

template <typename T>
std::string to_string (const T& t)
{
    std::stringstream ss;
    ss << t;
    return ss.str();
}

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
    if (V[i]!=i) std::cout <<  std::setw(2) << to_string<long>(mapping[i]) << " ";
    else std::cout << " -" << " ";
  std::cout << std::endl;
}

static void reset(long V[], long size) {
  for (long i=0;i<size;++i) V[i] =i;
}


// All parfors inc element-wise the array A.

int main() {
  long nworkers = 2;
  long A[N];
  for (long i=0;i<N;++i) A[i]=i;


  /* Version 1 (basic)

     void parallel_for (long first, long last, const Function &f, const long nw=FF_AUTO)

    Compute the body on [first,last) elements
    Static scheduling. grain = N/nw
  */

#ifdef FF
  ff::ParallelFor pf(nworkers, false);
  pf.parallel_for(0L,N,[&A](const long i) {
      A[i]+=1;
    });
#else // OMP
#pragma omp parallel for num_threads(nworkers)
  for(long i=0;i<N;++i) {
      A[i]+=1;
    };
#endif
  
  std::cout << "====================================================" << std::endl;
  std::cout << "1) Basic" << std::endl;
  print(A,N);
  compute_mask(A,N," X");
  reset(A,N);

  /* Version 2 (step)
     
     parallel_for (long first, long last, long step, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Static scheduling, grain = (last-first)/(step*nw)

    X  -  X  -  -  X  -  X  -  -
    Error to be fixed

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

  /* Version 3 (step, grain)
     
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

  /* Version 4 (step, grain, threadID)
     
     parallel_for_thid (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Static scheduling, chunk defined by user, body can access to worker number
  */
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;

  pf.parallel_for_thid(0L,N,step,grain,[&A,&mapping_on_threads](const long i, const int thid) {
      A[i]+=1;
      mapping_on_threads[i] = thid;
    });

  std::cout << "====================================================" << std::endl;
  std::cout << "4) step= " << step << "grain = " << grain << 
    " with ThreadIDs" << std::endl;
  print(A,N);
  compute_mask_i(A,N,mapping_on_threads);
  reset(A,N);

   /* Version 5 (step, grain, IDX)
     
     parallel_for_idx (long first, long last, long step, long grain, const Function &f, const long nw=FF_AUTO)
    
    Compute the body on [first,last) elements with step
    Static scheduling, chunk defined by user, body can access to worker number

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

  /* ---------- */
  std::cout << "====================================================" << std::endl;
  step = 1;
  for (long i=0;i<N;++i) mapping_on_threads[i] =-1;
  pf.parallel_for_static(0L,N,step,0,[&A,&mapping_on_threads](const long i) {

      A[i]+=1;
      // Internal, OS-dependent thread ID
      mapping_on_threads[i] = ff_getThreadID(); 
    });

  std::cout << "6) Static with maximal balanced partitions" << std::endl; 
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
  pf.parallel_for_static(0L,N,step,grain,[&A,&mapping_on_threads](const long i) {
      A[i]+=1;
      // Internal, OS-dependent thread ID
      mapping_on_threads[i] = ff_getThreadID(); 
    });
  
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
	
  
