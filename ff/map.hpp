/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file map.hpp
 *  \ingroup high_level_patterns
 *
 *  \brief map pattern
 * 
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

 
#ifndef FF_MAP_HPP
#define FF_MAP_HPP

// NOTE: A better check would be needed !
// both GNU g++ and Intel icpc define __GXX_EXPERIMENTAL_CXX0X__ if -std=c++0x or -std=c++11 is used 
// (icpc -E -dM -std=c++11 -x c++ /dev/null | grep GXX_EX)
#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_AUTO) && defined(HAS_CXX11_LAMBDA))
#include <ff/parallel_for.hpp>
#else
#error "C++ >= 201103L is required to use ff_Map"
#endif

namespace ff {


/*!
 * \class Map pattern
 *  \ingroup high_level_patterns
 *
 * \brief Map pattern
 *
 * Apply to all
 *
 * \todo Map to be documented and exemplified
 */
template<typename S, typename reduceT=int>
class ff_Map: public ff_node_t<S>, public ParallelForReduce<reduceT> {
    using ParallelForReduce<reduceT>::pfr;
protected:
    int prepare() {
        if (!prepared) {
            // warmup phase
            pfr->resetskipwarmup();
            auto r=-1;
            if (pfr->run_then_freeze() != -1)         
                r = pfr->wait_freezing();            
            if (r<0) {
                error("ff_Map: preparing ParallelForReduce\n");
                return -1;
            }
            
            if (spinWait) { 
                if (pfr->enableSpinning() == -1) {
                    error("ParallelForReduce: enabling spinwait\n");
                    return -1;
                }
            }
            prepared = true;
        }
        return 0;
    }
        
    int freeze_and_run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;
        return ff_node::freeze_and_run(true);
    }
    
public:
    ff_Map(size_t maxp=-1, bool spinWait=false, bool spinBarrier=false):
        ParallelForReduce<reduceT>(maxp,false,true,false),// skip loop warmup and disable spinwait
        spinWait(spinWait),prepared(false)  {
        ParallelForReduce<reduceT>::disableScheduler(true);
    }

    int run(bool=false) {
        if (!prepared) if (prepare()<0) return -1;
        return ff_node::run(true);
    }

    int run_then_freeze() {
        return freeze_and_run();
    }

    int wait() { return ff_node::wait();}
    int wait_freezing() { return ff_node::wait_freezing(); }
protected:
    bool spinWait;
    bool prepared;
};
    
} // namespace ff

#endif /* FF_MAP_HPP */

