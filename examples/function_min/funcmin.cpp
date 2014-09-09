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
 * Finding the minimum in a given interval (min-range;max-range( of a function f_x 
 * (single variable) continuous in the range considered using the Pool Evolution 
 * parallel pattern.
 * 
 * Date: June 2014
 * Authors: 
 *      Sonia Campa      <campa@di.unipi.it>
 *      Marco Danelutto  <marco.danelutto@di.unipi.it>
 *      Massimo Torquati <torquati@di.unipi.it>
 *      
 */

#include <random>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <cstdint>
#include <vector>
#include <cmath>
#include <ff/poolEvolution.hpp>

// basic configuration
#define PARALLEL_SELECTION 1
#define PARALLEL_FILTERING 1
#undef  UNIQUE_RANDOM_NUM 

using namespace ff;
using namespace std;


// this is the function
// it has to be continuous inside the interval considered 
static inline double f_x(const double x){
    return pow(x,12) - pow(x,8) - pow(x,5) + x/2 - 1.14;  // the minimum is in x=1.008337799
}

// a pair <value, f_value>
struct Couple {
    Couple() {}
    Couple(double v):value(v) {}
    double f_value; 
    double value;

    bool operator() (double i,double j) { return (i<j);}
    bool operator== (const Couple& c) { return (c.value==value); }
};
static inline bool sortobj(const Couple& a,const Couple& b){ 
    return(a.f_value<b.f_value);
}


// butta
void printPop(const char *str, std::vector<Couple> &P) {
    printf("%s:\n", str);
    for(size_t i=0;i<P.size();++i)
        printf("(%g,%g) ", P[i].value,P[i].f_value);
    printf("\n");
}



// 'global' environment
struct Env_t {
    long   Epardegree;
    long   Spardegree;
    long   Fpardegree;
    long   local_iter;
    long   global_iter;
    long   num_param;
    long   numSurvivors;
    double  min_range; 
    double  max_range;
    double  delta;
};

// this contains all the randum numbers needed during the computation
// they are pre-computed at the very beginning (not space-efficient, indeed) 
std::vector<double> random_nums;

/*
 *    Utility function
 */
static void buildPopulation(long numparam, std::vector<Couple>& P, const Env_t& env) {
    for (long i=0;i<numparam;i++){
      Couple f;
      assert(random_nums.size()>0);
      f.value = random_nums.back();
      random_nums.pop_back();
      f.f_value = f_x(f.value);
      P.push_back(f);
    }
}

/*======================================================================*/

/*
 *    Pool's functions
 */

/*----------------------------------------------------*/
bool termination(const std::vector<Couple> &P, Env_t &env) {
    env.global_iter--;
    if (env.global_iter < 0) return true; 
    return false;
}

/*----------------------------------------------------*/
// identity function
void selection(ParallelForReduce<Couple>& pfr, std::vector<Couple>& P, std::vector<Couple>& buffer,Env_t &env) {
#if defined(PARALLEL_SELECTION)
  buffer.resize(P.size());
  auto copy = [&](const long i) {  buffer[i]=P[i];  };
  pfr.parallel_for(0,P.size(),copy);
#else
  buffer.insert( buffer.end(), P.begin(), P.end());
#endif
}

/*----------------------------------------------------*/
const Couple& evolution(Couple& individual, const Env_t& env,const int) {
  double dx=individual.f_value;
  double sx=individual.f_value;
  for (long i=0;i< env.local_iter;i++){
      double newdx = individual.value+env.delta;
      double newsx = individual.value-env.delta;
      if (newdx<=env.max_range)
          dx = f_x(newdx);
      if (newsx>=env.min_range)
          sx = f_x(newsx);
      if (dx<individual.f_value){
          individual.value = newdx;
          individual.f_value = dx;
      }
      if (sx<individual.f_value){
          individual.value = newsx;
          individual.f_value = sx;
      }
  }
  return individual;
}

/*----------------------------------------------------*/
static inline void crossover (std::vector<Couple>::iterator b, 
			      std::vector<Couple>::iterator e) {  
    std::vector<Couple>::iterator k = b;
    while(b < e) {
        double  son = ( (*b).value+ (*(b+1)).value )/2;
        (*k).value   = son;
        (*k).f_value = f_x(son);
        k++;
        b+=2;
    }  
}
/*----------------------------------------------------*/

void filter(ParallelForReduce<Couple>& pfr, 
	    std::vector<Couple>& P, std::vector<Couple>& buffer,Env_t& env) {
#if defined(PARALLEL_FILTERING)
    const long K = (env.numSurvivors/env.Fpardegree)/2;
    auto F = [&](const long start, const long stop, const int thid) {  
        if (start==stop) return;
        std::vector<Couple> tmp;
        tmp.insert(tmp.end(), buffer.begin()+start, buffer.begin()+stop);
        std::sort(tmp.begin(), tmp.end(), sortobj);
        std::vector<Couple>::iterator b = tmp.begin()+1;
        std::vector<Couple>::iterator e = tmp.begin()+K+1;
        crossover(b, e);
        tmp.erase(tmp.begin()+K+2, tmp.end());
        const long size = (stop-start)-(K+2);
        long idx = env.global_iter*env.numSurvivors + K*thid;
        for(long i=0;i<size;++i) {
            Couple cp;
            cp.value = random_nums[idx++];
            cp.f_value = f_x(cp.value);
            tmp.push_back(std::move(cp));
        }
        assert(tmp.size() == static_cast<size_t>((stop-start)));
        for(size_t i=0;i<tmp.size();++i)
            buffer[start+i] = std::move(tmp[i]);
    };
    pfr.parallel_for_idx(0, buffer.size(), 1, 0, F,env.Fpardegree);
#else
    std::sort (buffer.begin(), buffer.end(),sortobj);
    const long K = (env.numSurvivors/2);
    crossover(buffer.begin()+1, buffer.begin()+K+1);
    buffer.erase(buffer.begin()+K+2, buffer.end());    

    for (long i=0;i<env.num_param-(K+1);i++){
        assert(random_nums.size()>0);
        Couple cp;
        cp.value = random_nums.back();  
        random_nums.pop_back();
        cp.f_value = f_x(cp.value);
        buffer.push_back(std::move(cp));
    }
#endif
}

/*======================================================================*/

int main(int argc, char * argv[]) {
  Env_t env;
  if (argc>=9){
    env.global_iter  = atol(argv[1]);
    env.num_param    = atol(argv[2]);
    env.numSurvivors = atol(argv[3]);
    env.min_range    = atof(argv[4]);
    env.max_range    = atof(argv[5]);
    env.local_iter   = atoi(argv[6]);
    env.delta        = atof(argv[7]);
    env.Epardegree   = atol(argv[8]);
    env.Spardegree   = 0;
    env.Fpardegree   = 0;
    if (argc==10) {
        env.Spardegree = atol(argv[9]);
        env.Fpardegree = env.Spardegree;
    }
    if (argc==11) {
        env.Spardegree = atol(argv[9]);
        env.Fpardegree = atol(argv[10]);
    }

    cout << "Param used:\n";
    cout << " global iterations:\t" << env.global_iter  << "\n";
    cout << " n. of points     :\t" << env.num_param << "\n";
    cout << " n. survivors     :\t" << env.numSurvivors << "\n";
    cout << " range (min,max(  :\t" << "(" << env.min_range << "," << env.max_range << "(\n";
    cout << " delta            :\t" << env.delta << "\n";
    cout << " internal iter    :\t" << env.local_iter << "\n";
    cout << " pardegree E,S,F  :\t" << env.Epardegree << "," << env.Spardegree << "," << env.Fpardegree << "\n";
    cout << "\n\n";
  }else{
      cout<<"params: glob_iterations num_points num_survivors min_range max_range local_iter delta EvolTh";
#if PARALLEL_SELECTION
      cout << " SelecTh";
#endif
#if PARALLEL_FILTERING
      cout << " FiltTh";
#endif
      cout << "\n";
      cout<< "   " << argv[0] << " 20 10000 5000 -1.5 1.5 50 0.000001 24";
#if PARALLEL_SELECTION
      cout << " 12";
#endif
#if PARALLEL_FILTERING
      cout << " 48";
#endif
      cout << "\n";
      return -1;
  }

  // basic checks
  if ((abs(env.max_range - env.min_range))/env.delta < env.num_param)  {
      std::cerr << "ERROR: range too small, reduce delta or enlarge the point range\n\n";
      return -1;
  }
#if defined(PARALLEL_FILTER)
  if (env.numSurvivors > ((env.num_param / 2) + env.Epardegree)) {
      std::cerr << "ERROR: too many survivors, num_survivors should be less than (num_points/2 + Epardegree)\n\n";
      return -1;
  }
  // generating all random numbers needed beforehand
  const long num_randoms = (env.global_iter)*(env.num_param-(env.numSurvivors/2+env.Epardegree))+env.num_param;
#else
  if (env.numSurvivors > (env.num_param / 2)) {
      std::cerr << "ERROR: too many survivors, num_survivors should be less than num_points/2\n\n";
      return -1;
  }


  // generating all random numbers needed beforehand
  const long num_randoms = (env.global_iter)*(env.num_param-(env.numSurvivors/2+1))+env.num_param;
#endif


  random_nums.resize(num_randoms);
  std::random_device rd;
  //std::default_random_engine generator(rd());
  std::default_random_engine generator;
  std::uniform_real_distribution<double> distribution(0.0,1.0);
  const double range = fabs(env.max_range-env.min_range);

  for(long i=0;i<num_randoms;++i) {
      double value;
      value = distribution(generator)*range + env.min_range;
#if defined(UNIQUE_RANDOM_NUM)
      //
      // WARNING: it takes lots of time for generating unique numbers and it is not really needed....
      //
  retry:
      if (std::find(random_nums.begin(),random_nums.end(), value) != random_nums.end()) {
	  value+=distribution(generator);
	  goto retry;
      }
#endif
      random_nums[i] = value;
  }

  std::vector<Couple> P;
  buildPopulation(env.num_param, P, env);
  assert(random_nums.size() == static_cast<size_t>(num_randoms-env.num_param));


  poolEvolution<Couple, Env_t> pool(std::max(env.Epardegree,std::max(env.Spardegree,env.Fpardegree)), 
                                    P, selection,evolution,filter,termination, env);
  
  ffTime(START_TIME);
  if (pool.run_and_wait_end()<0)
      abort();
  ffTime(STOP_TIME);

  std::cout << "DONE, total time= " << ffTime(GET_TIME) << " (ms)\n";
  std::cout << "Best min (x,y) = (" << std::setprecision(15) << P[0].value << ", " << std::setprecision(15) << P[0].f_value << ")\n";
  
  return 0;
}
