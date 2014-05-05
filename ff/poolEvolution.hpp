/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file poolEvolution.hpp
 *  \ingroup high_level_patterns_shared_memory
 *
 *  \brief This file describes the pool evolution pattern.
 */
 
/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
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

#ifndef FF_POOL_HPP
#define FF_POOL_HPP

#include <vector>
#include <ff/node.hpp>
#include <ff/parallel_for.hpp>

namespace ff {

template<typename T>
class poolEvolution : public ff_node {
public:
    typedef typename std::vector<T>::iterator       iter_t;
    typedef typename std::vector<T>::const_iterator const_iter_t;
protected:
    size_t maxp,pE,pF,pT,pS;
    std::vector<T>               *input;
    std::vector<T>                buffer;
    std::vector<std::vector<T> >  bufferPool;
    void (*selection)(const_iter_t start, const_iter_t stop, std::vector<T> &out);
    const T& (*evolution)(T& individual);
    void (*filter)(const_iter_t start, const_iter_t stop, std::vector<T> &out);
    bool (*termination)(const std::vector<T> &pop); 
    ParallelForReduce<T> loopevol;
public :
    // constructor : to be used in non-streaming applications
    poolEvolution (size_t maxp,                                // maximum parallelism degree 
                   std::vector<T> & pop,                       // the initial population
                   void (*sel)(const_iter_t start, const_iter_t stop,
                               std::vector<T> &out),           // the selection function
                   const T& (*evol)(T& individual),            // the evolution function
                   void (*fil)(const_iter_t start, const_iter_t stop,
                               std::vector<T> &out),           // the filter function
                   bool (*term)(const std::vector<T> &pop))    // the termination function
        :maxp(maxp), pE(maxp),pF(1),pT(1),pS(1),input(&pop),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp) { 
        loopevol.disableScheduler(true);
    }
    // constructor : to be used in streaming applications
    poolEvolution (size_t maxp,                                // maximum parallelism degree 
                   void (*sel)(const_iter_t start, const_iter_t stop,
                               std::vector<T> &out),           // the selection function
                   const T& (*evol)(T& individual),            // the evolution function
                   void (*fil)(const_iter_t start, const_iter_t stop,
                               std::vector<T> &out),        // the filter function
                   bool (*term)(const std::vector<T> &pop))    // the termination function
        :maxp(maxp), pE(maxp),pF(1),pT(1),pS(1),input(NULL),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp) { 
        loopevol.disableScheduler(true);
    }
    
    // the function returning the result in non streaming applications
    const std::vector<T>& get_result() { return *input; }
    
    // changing the parallelism degree
    void setParEvolution(size_t pardegree)   { 
        if (pardegree>maxp)
            error("setParEvolution: pardegree too high, it should be less than or equal to %ld\n",maxp);
        else pE = pardegree;        
    }
    // currently the termination condition is computed sequentially
    void setParTermination(size_t )          { pT = 1; }
    void setParSelection(size_t pardegree)   { 
        if (pardegree>maxp)
            error("setParSelection: pardegree too high, it should be less than or equal to %ld\n",maxp);
        else pS = pardegree;
    }
    void setParFilter (size_t pardegree)     { 
        if (pardegree>maxp)
            error("setParFilter: pardegree too high, it should be less than or equal to %ld\n",maxp);
        else pS = pardegree;
    }
        
    int run_and_wait_end() {
        // TODO:
        // if (isfrozen()) {
        //     stop();
        //     thaw();
        //     if (wait()<0) return -1;
        //     return 0;
        // }
        // stop();
        if (ff_node::run()<0) return -1;
        if (ff_node::wait()<0) return -1;
        return 0;
    }
    
protected:
    void* svc(void * task) {
        if (task) input = ((std::vector<T>*)task);

        size_t max_pS_pF = std::max(pS,pF);
        bufferPool.resize(max_pS_pF);
        
        while(!termination(*input)) {
            // selection phase
            buffer.clear();
            if (pS==1) 
                selection(input->begin(),input->end(), buffer);
            else {
                for(size_t i=0;i<pS;++i) bufferPool[i].clear();

                auto S = [&](const long start, const long stop, const int thread_id) {
                    printf("Selection: %d (%ld,%ld(\n", thread_id, start, stop);

                    auto begin = input->begin()+start;
                    auto end   = input->begin()+start+stop;
                    selection(begin,end, bufferPool[thread_id]);
                };
                loopevol.parallel_for_idx(0,input->size(),1,-1,S,pS); // TODO: static and dynamic scheduling

                for(size_t i=0;i<pS;++i)
                    buffer.insert( buffer.end(), bufferPool[i].begin(), bufferPool[i].end());
            }

            // evolution phase
            auto E = [&](const long i) {
                buffer[i]=evolution(buffer[i]); 
            };
            loopevol.parallel_for(0,buffer.size(),E, pE); // TODO: static and dynamic scheduling

            // filtering phase
            input->clear();
            if (pF==1)
                filter(buffer.begin(),buffer.end(), *input);
            else {
                for(size_t i=0;i<pF;++i) bufferPool[i].clear();

                auto F = [&](const long start, const long stop, const int thread_id) {
                    auto begin = buffer.begin()+start;
                    auto end   = buffer.begin()+start+stop;
                    filter(begin, end, bufferPool[thread_id]);
                };
                loopevol.parallel_for_idx(0,buffer.size(),1,-1,F,pF); // TODO: static and dynamic scheduling
                for(size_t i=0;i<pF;++i)
                    input->insert( input->end(), bufferPool[i].begin(), bufferPool[i].end());
            }
        }
        return (task?input:NULL);
    }
};
    
} // namespace ff


#endif /* FF_POOL_HPP */
