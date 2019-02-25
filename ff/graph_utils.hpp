/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \file graph_utils.hpp
 * \ingroup aux_classes
 * \brief Utility functions for manipulating the concurrency graph
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
 *
 ****************************************************************************
 */

#ifndef FF_GRAPH_UTILS_HPP
#define FF_GRAPH_UTILS_HPP

/*
 *   Author: Massimo Torquati
 *      
 */


#include <ff/svector.hpp>
#include <ff/node.hpp>
#include <ff/farm.hpp>
#include <ff/all2all.hpp>
#include <ff/pipeline.hpp>

namespace ff {

static svector<ff_node*> empty_svector(0);    
    
// checks if node is a farm with the collector node 
static inline bool isfarm_withcollector(ff_node* node) {
    if (node->isFarm()) {
        ff_farm* farm= reinterpret_cast<ff_farm*>(node);
        return farm->hasCollector();
    }
    return false;
}
// checks if node is a farm with no collector and multi-output workers
static inline bool isfarm_multimultioutput(ff_node* node) {
    if (node->isFarm()) {
        ff_farm* farm= reinterpret_cast<ff_farm*>(node);
        if (farm->hasCollector()) return false;
        const svector<ff_node*>& w1 = farm->getWorkers();
        if (w1[0]->isMultiOutput()) return true;   // NOTE: here we suppose homogeneous workers
    }
    return false;
}
// checks if node is a farm, if yes it returns its worker nodes 
static inline const svector<ff_node*>& isfarm_getworkers(ff_node* node) {
    if (node->isFarm()) {
        ff_farm* farm= reinterpret_cast<ff_farm*>(node);
        return farm->getWorkers();
    }
    return empty_svector;
}
// checks if node is an all-to-all, if yes it returns the first set
static inline const svector<ff_node*>& isa2a_getfirstset(ff_node* node) {
    if (node->isAll2All()) {
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(node);
        return a2a->getFirstSet();
    }
    return empty_svector;
}
// checks if node is an all-to-all, if yes it returns the second set
static inline const svector<ff_node*>& isa2a_getsecondset(ff_node* node) {
    if (node->isAll2All()) {
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(node);
        return a2a->getSecondSet();
    }
    return empty_svector;
}
// checks if node is a pipeline, if yes it returns the last stage 
static inline ff_node* ispipe_getlast(ff_node* node) {
    if (node->isPipe()) {
        ff_pipeline* pipe = reinterpret_cast<ff_pipeline*>(node);
        return pipe->get_lastnode();
    }
    return nullptr;
}
    
} // namespace
#endif /* FF_GRAPH_UTILS_HPP */
