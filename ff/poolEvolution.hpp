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
protected:
    size_t pE,pF,pT,pS;
    std::vector<T>  *input;
    std::vector<T>   buffer;
    void (*selection)(const std::vector<T> &pop,std::vector<T> &out);
    const T& (*evolution)(T& individual);
    void (*filter)(const std::vector<T> &pop,std::vector<T> &out);
    bool (*termination)(const std::vector<T> &pop); 
    ParallelForReduce<T> loopevol;
public :
    // constructor : to be used in non-streaming applications
    poolEvolution (size_t maxp,                                // maximum parallelism degree of the evolution phase 
                   std::vector<T> & pop,                       // the initial population
                   void (*sel)(const std::vector<T> &pop, 
                               std::vector<T> &out),           // the selection function
                   const T& (*evol)(T& individual ),           // the evolution function
                   void (*fil)(const std::vector<T> &pop,
                               std::vector<T> &out),           // the filter function
                   bool (*term)(const std::vector<T> &pop))    // the termination function
        :pE(maxp),pF(1),pT(1),pS(1),input(&pop),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp) { 
        loopevol.disableScheduler(true);
    }
    // constructor : to be used in streaming applications
    poolEvolution (size_t maxp,                                // parallelism degree 
                   void (*sel)(const std::vector<T> & pop, 
                               std::vector<T> &out),        // the selection function
                   const T& (*evol)(T& individual ),           // the evolution function
                   void (*fil)(const std::vector<T> &pop,
                               std::vector<T> &out),        // the filter function
                   bool (*term)(const std::vector<T> &pop))    // the termination function
        :pE(maxp),pF(1),pT(1),pS(1),input(NULL),selection(sel),evolution(evol),filter(fil),termination(term),
         loopevol(maxp) { 
        loopevol.disableScheduler(true);
    }
    
    // the function returning the result in non streaming applications
    const std::vector<T>& get_result() { return *input; }
    
    // changing the parallelism degree
    void setParEvolution(size_t pardegree)   { pE = pardegree; }
    void setParTermination(size_t pardegree) { pT = pardegree; }
    void setParSelection(size_t pardegree)   { pS = pardegree; }
    void setParFilter (size_t pardegree)     { pF = pardegree; }
    
    
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
        
        while(!termination(*input)) {
            buffer.clear();
            selection(*input, buffer);	    
            auto E = [&](const long i) {
                buffer[i]=evolution(buffer[i]); 
            };
            loopevol.parallel_for(0,buffer.size(),E, pE);
            input->clear();
            filter(buffer, *input);
        }
        return (task?input:NULL);
    }
};
    
} // namespace ff


#endif /* FF_POOL_HPP */
