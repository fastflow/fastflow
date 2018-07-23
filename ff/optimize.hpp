/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 * \link
 * \file optimize.hpp
 * \ingroup building_blocks
 *
 * \brief FastFlow optimization heuristics 
 *
 * @detail FastFlow basic contanier for a shared-memory parallel activity 
 *
 */

#ifndef FF_OPTIMIZE_HPP
#define FF_OPTIMIZE_HPP

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

/*
 *   Author: Massimo Torquati
 *      
 */
#include <cstdio>
#include <cstdlib>
#include <cstdarg>
#include <ff/node.hpp>
#include <ff/pipeline.hpp>
#include <ff/farm.hpp>
#include <ff/all2all.hpp>
#include <ff/combine.hpp>

namespace ff {

typedef enum { OPT_NORMAL = 1, OPT_INFO = 2 } reportkind_t;
static inline void opt_report(int verbose_level, reportkind_t kind, const char *str, ...) {
    if (verbose_level < kind) return;
    va_list argp;
    char * p=(char *)malloc(strlen(str)+512); // this is dangerous.....
    assert(p);
    strcpy(p, str);
    va_start(argp, str);
    vfprintf(stdout, p, argp);
    va_end(argp);
    free(p);
}
    
/* This is farm specific. 
 *  - It basically sets the threshold for enabling blocking mode.
 *  - It can remove the collector of internal farms
 *  - TODO: Farm of farms?????? ---> single farm with two emitters combined (external+internal)
 */    
int optimize_static(ff_farm& farm, const OptLevel& opt=OptLevel1()) {
    if (farm.prepared) {
        error("optimize_static (farm) called after prepare\n");
        return -1;
    }
    // swithing to blocking mode if the n. of threads is greater than the threshold
    if (opt.blocking_mode) {
        ssize_t card = farm.cardinality();
        if (opt.max_nb_threads < card) {
            opt_report(opt.verbose_level, OPT_NORMAL,
                       "OPT (farm): BLOCKING_MODE: Activating blocking mode, threshold=%ld, number of threads=%ld\n",opt.max_nb_threads, card);
            
            farm.blocking_mode(true);
        }
    }
    // turning off initial/default mapping if the n. of threads is greater than the threshold
    if (opt.no_default_mapping) {
        ssize_t card = farm.cardinality();
        if (opt.max_mapped_threads < card) {
            opt_report(opt.verbose_level, OPT_NORMAL,
                       "OPT (farm): MAPPING: Disabling mapping, threshold=%ld, number of threads=%ld\n",opt.max_mapped_threads, card);
            
            farm.no_mapping();
        }
    }
    // here it looks for internal farms with null collectors
    if (opt.remove_collector) {
        const svector<ff_node*>& W = farm.getWorkers();
        for(size_t i=0;i<W.size();++i) {
            if (W[i]->isFarm() && !W[i]->isOFarm()) {
                ff_farm* ifarm = reinterpret_cast<ff_farm*>(W[i]);
                OptLevel iopt;
                iopt.remove_collector=true;
                iopt.verbose_level = opt.verbose_level;
                if (optimize_static(*ifarm, iopt)<0) return -1;
                if (ifarm->getCollector() == nullptr) {
                    opt_report(opt.verbose_level, OPT_NORMAL, "OPT (farm): REMOVE_COLLECTOR: Removed farm collector\n");
                    ifarm->remove_collector();
                }
            }
        }
    }
    // no initial barrier
    if (opt.no_initial_barrier) {
        opt_report(opt.verbose_level, OPT_NORMAL,
                   "OPT (farm): NO_INITIAL_BARRIER: Initial barrier disabled\n");
        farm.no_barrier();
   }
    return 0;
}
    
/* 
 *
 */    
int optimize_static(ff_pipeline& pipe, const OptLevel& opt=OptLevel1()) {
    if (pipe.prepared) {
        error("optimize_static (pipeline) called after prepare\n");
        return -1;
    }
   // flattening the pipeline ----------------------------------
   const svector<ff_node*>& W = pipe.get_pipeline_nodes();
   int nstages=static_cast<int>(pipe.nodes_list.size());
   for(int i=0;i<nstages;++i)  pipe.remove_stage(0); 
   nstages = static_cast<int>(W.size());
   for(int i=0;i<nstages;++i) pipe.add_stage(W[i]);
   // ----------------------------------------------------------

   // ------------------ helping function ----------------------
   auto find_farm_with_null_collector =
       [](const svector<ff_node*>& nodeslist, int start=0)->int {
       for(int i=start;i<(int)(nodeslist.size());++i) {
           if (nodeslist[i]->isFarm()) {
               ff_farm *farm = reinterpret_cast<ff_farm*>(nodeslist[i]);
               if (farm->getCollector() == nullptr)  return i;
           }
       }
       return -1;
   };
   auto find_farm_with_null_emitter =
       [](const svector<ff_node*>& nodeslist, int start=0)->int {
       for(int i=start;i<(int)(nodeslist.size());++i) {
           if (nodeslist[i]->isFarm() &&
               (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getEmitter()) 
               ) 
               return i;
       }
       return -1;
   };
   // looking for the longest sequence of farms (or ofarms) with the same number of workers
   auto farm_sequence =
       [&](const svector<ff_node*>& nodeslist, int& first_farm, int& last_farm) {
       bool ofarm = nodeslist[first_farm]->isOFarm();
       size_t nworkers = (reinterpret_cast<ff_farm*>(nodeslist[first_farm]))->getNWorkers();
       int starting_point=first_farm+1;
       int first = first_farm, last = last_farm;
       while (starting_point<static_cast<int>(nodeslist.size()))  {
           bool ok=true;
           int next = find_farm_with_null_emitter(nodeslist, starting_point);
           if (next == -1) break;
           else {
               for(int i=starting_point; i<=next;) {
                   if ((ofarm?nodeslist[i]->isOFarm():!nodeslist[i]->isOFarm()) &&
                       (nworkers == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getNWorkers())  &&
                       (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getCollector()) &&
                       (nullptr  == (reinterpret_cast<ff_farm*>(nodeslist[i]))->getEmitter())
                       )
                       ++i;
                   else { ok = false; break; }
               }
               if (ok) 
                   if ((reinterpret_cast<ff_farm*>(nodeslist[next]))->getEmitter() != nullptr) ok=false;
           }
           if (ok) {
               last = next;
               starting_point = next+1;
           } else {
               if (last==-1) {
                   first = find_farm_with_null_collector(nodeslist, first+1);
                   if (first==-1) break;
                   starting_point = first_farm+1;
               } break;
           }                   
       }
       if (first != -1 && last != -1) {
           first_farm = first;
           last_farm  = last;
       }
   };
   // introduces the normal-form of a sequence of farms (or ofarms)
   auto combine_farm_sequence =
       [](const svector<ff_node*>& nodeslist, int first_farm, int last_farm) {

       svector<svector<ff_node*> > W(16);
       W.resize(last_farm-first_farm+1);
       for(int i=first_farm, j=0; i<=last_farm; ++i,++j) {
           W[j]=reinterpret_cast<ff_farm*>(nodeslist[i])->getWorkers();
       }
       size_t nfarms   = W.size();
       size_t nworkers = W[0].size();
       
       std::vector<ff_node*> Workers(nworkers);
       for(size_t j=0; j<nworkers; ++j) {
           if (nfarms==2)  {
               ff_comb *p = new ff_comb(W[0][j],W[1][j]);
               assert(p);
               Workers[j] = p;
           } else {
               const ff_comb *p = new ff_comb(W[0][j],W[1][j]);
               for(size_t i=2;i<nfarms;++i) {
                   const ff_comb* combtmp = new ff_comb(*p, W[i][j]);
                   assert(combtmp);
                   delete p;
                   p = combtmp;                           
               }
               Workers[j] = const_cast<ff_comb*>(p);
           }
       }
       ff_farm* firstfarm= reinterpret_cast<ff_farm*>(nodeslist[first_farm]);
       ff_farm* lastfarm = reinterpret_cast<ff_farm*>(nodeslist[last_farm]);
       ff_farm* newfarm = new ff_farm;
       if (firstfarm->isOFarm()) {
           assert(lastfarm->isOFarm());
           newfarm->set_ordered();
       }
       
       if (firstfarm->getEmitter()) newfarm->add_emitter(firstfarm->getEmitter());       
       if (lastfarm->hasCollector()) 
           newfarm->add_collector(lastfarm->getCollector());
       newfarm->add_workers(Workers);
       newfarm->cleanup_workers();
       newfarm->set_scheduling_ondemand(firstfarm->ondemand_buffer());
       
       return newfarm;
   };
   // ---------------- end helping function --------------------
       
   if (opt.merge_farms) {
       // find the first farm with default collector or with no collector
       int first_farm = find_farm_with_null_collector(pipe.nodes_list);
       if (first_farm!=-1) {
           do {
               int last_farm  = -1;
               farm_sequence(pipe.nodes_list,first_farm,last_farm);
               if (first_farm<last_farm) {  // normal form
                   ff_farm *newfarm = combine_farm_sequence(pipe.nodes_list,first_farm,last_farm);               
                   for(int i=first_farm; i<=last_farm; ++i) pipe.remove_stage(first_farm);
                   pipe.insert_stage(first_farm, newfarm, true);
                   opt_report(opt.verbose_level, OPT_NORMAL,
                              "OPT (pipe): MERGE_FARMS: Merged farms staged [%d-%d]\n", first_farm, last_farm);

               } 
               first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
           }while(first_farm!=-1 && first_farm < static_cast<int>(pipe.nodes_list.size()));
       }
   }
   if (opt.introduce_a2a) {
       int first_farm = find_farm_with_null_collector(pipe.nodes_list);
       while(first_farm != -1 && (first_farm < static_cast<int>(pipe.nodes_list.size()-1))) {
           if (!pipe.nodes_list[first_farm]->isOFarm()) {
               if (pipe.nodes_list[first_farm+1]->isFarm() && !pipe.nodes_list[first_farm+1]->isOFarm()) {
                   ff_farm *farm1 = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                   ff_farm *farm2 = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm+1]);
                   if (farm2->getEmitter() == nullptr) {
                       opt_report(opt.verbose_level, OPT_NORMAL,
                                  "OPT (pipe): INTRODUCE_A2A: Introducing all-to-all between %d and %d stages\n", first_farm, first_farm+1);
                       const ff_farm f = combine_farms_a2a(*farm1, *farm2);
                       ff_farm* newfarm = new ff_farm(f);
                       assert(newfarm);
                       pipe.remove_stage(first_farm);
                       pipe.remove_stage(first_farm);
                       pipe.insert_stage(first_farm, newfarm, true);
                       first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
                   }
               } else {
                   if (pipe.nodes_list[first_farm+1]->isOFarm())
                       opt_report(opt.verbose_level, OPT_INFO,
                                  "OPT (pipe): INTRODUCE_A2A: cannot introduce A2A because node %d is an ordered farm\n", first_farm+1);
                   first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+2);
               }
           } else {
               opt_report(opt.verbose_level, OPT_INFO,
                          "OPT (pipe): INTRODUCE_A2A: cannot introduce A2A because node %d is an ordered farm\n", first_farm);
               first_farm = find_farm_with_null_collector(pipe.nodes_list, first_farm+1);
           }
       }
   }

   if (opt.remove_collector) {
       // first, for all farms in the pipeline we try to optimize farms' workers
       OptLevel farmopt;
       farmopt.remove_collector = true;
       farmopt.verbose_level = opt.verbose_level;
       for(size_t i=0;i<pipe.nodes_list.size();++i) {
           if (pipe.nodes_list[i]->isFarm()) {
               ff_farm *farmnode = reinterpret_cast<ff_farm*>(pipe.nodes_list[i]);
               if (optimize_static(*farmnode, farmopt)<0) {
                   error("optimize_static, trying to optimize the farm at stage %ld\n", i);
                   return -1;
               }
           }
       }
       
       int first_farm = find_farm_with_null_collector(pipe.nodes_list);
       if (first_farm!=-1) {
           if (first_farm < static_cast<int>(pipe.nodes_list.size()-1)) {

               // TODO: if the next stage is A2A would be nice to have a rule
               //       that attaches the farm workers with the first set of nodes

               ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
               if (farm->isOFarm()) {
                   if ((!pipe.nodes_list[first_farm+1]->isAll2All()) &&
                       (!pipe.nodes_list[first_farm+1]->isFarm())) {
                       farm->add_collector(pipe.nodes_list[first_farm+1]);
                       pipe.remove_stage(first_farm+1);
                       opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Merged next stage with ordered-farm collector\n");
                   }
               } else {
                   if (!pipe.nodes_list[first_farm+1]->isAll2All()) {
                       farm->remove_collector();
                       opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Removed farm collector\n");
                       
                       if (!pipe.nodes_list[first_farm+1]->isMultiInput()) {
                           // the next stage is a standard node
                           ff_node *next = pipe.nodes_list[first_farm+1];
                           pipe.remove_stage(first_farm+1);
                           ff_minode *mi = new mi_transformer(next);
                           assert(mi);
                           pipe.insert_stage(first_farm+1, mi, true);
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): REMOVE_COLLECTOR: Transformed next stage to multi-input node\n");
                       } 
                   }
               }
           }
       }
   }
   if (opt.merge_with_emitter) {
       int first_farm = find_farm_with_null_emitter(pipe.nodes_list);
       if (first_farm!=-1) {
           if (first_farm>0) { // it is not the first one
               bool prev_single_standard  = (!pipe.nodes_list[first_farm-1]->isMultiOutput());
               if (prev_single_standard) {
                   // could be a farm with a collector
                   if (pipe.nodes_list[first_farm-1]->isFarm()) {
                       ff_farm *farm_prev = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm-1]);
                       if (farm_prev->hasCollector()) {
                           ff_node* collector=farm_prev->getCollector();
                           if (collector->isMultiInput() && !collector->isComp()) {
                               error("MERGING MULTI-INPUT COLLECTOR TO THE FARM EMITTER NOT YET SUPPORTED\n");
                               abort();
                               // TODO we have to create a multi-input comp to add to the emitter
                           }
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with farm emitter\n");
                           ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                           farm->add_emitter(collector);
                           farm_prev->remove_collector();
                       }
                   } else {
                       ff_node *node = pipe.nodes_list[first_farm-1];
                       if (node->isMultiInput() && !node->isComp()) {
                           error("MERGING MULTI-INPUT NODE TO THE FARM EMITTER NOT YET SUPPORTED\n");
                           //TODO: we have to create a multi-input comp to add to the emitter
                           abort();
                       }
                       ff_farm *farm = reinterpret_cast<ff_farm*>(pipe.nodes_list[first_farm]);
                       if (pipe.nodes_list[first_farm]->isOFarm()) {
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with ordered-farm emitter\n");
                       } else {
                           opt_report(opt.verbose_level, OPT_NORMAL, "OPT (pipe): MERGE_WITH_EMITTER: Merged previous stage with farm emitter\n");                          
                       }
                       farm->add_emitter(node);
                       pipe.remove_stage(first_farm-1);
                   }
               }
           }
       }
   }
   
   // activate blocking mode if the n. of threads is greater than the threshold
   if (opt.blocking_mode) {
       ssize_t card = pipe.cardinality();
       if (opt.max_nb_threads < card) {
           opt_report(opt.verbose_level, OPT_NORMAL, 
                      "OPT (pipe): BLOCKING_MODE: Activating blocking mode, threshold=%ld, number of threads=%ld\n",opt.max_nb_threads, card);
           pipe.blocking_mode(true);
       } 
   }
    // turning off initial/default mapping if the n. of threads is greater than the threshold
   if (opt.no_default_mapping) {
       ssize_t card = pipe.cardinality();
       if (opt.max_mapped_threads < card) {
           opt_report(opt.verbose_level, OPT_NORMAL,
                      "OPT (pipe): MAPPING: Disabling mapping, threshold=%ld, number of threads=%ld\n",opt.max_mapped_threads, card);
           
           pipe.no_mapping();
       }
   }
   // no initial barrier
   if (opt.no_initial_barrier) {
       opt_report(opt.verbose_level, OPT_NORMAL,
                  "OPT (pipe): NO_INITIAL_BARRIER: Initial barrier disabled\n");
       pipe.no_barrier();
   }
   return 0;
}


#if 0    
template<typename T1, typename T2>
const ff_Pipe<typename T1::in_type, typename T2::out_type> optimize_static() {
    using pipe_t = ff_Pipe<typename T1::in_type, typename T2::out_type>;
    struct emptyNode:ff_node_t<typename T1::in_type, typename T2::out_type> {
        using in_type  = typename ff_node_t<typename T1::in_type, typename T2::out_type>::in_type;
        using out_type = typename ff_node_t<typename T1::in_type, typename T2::out_type>::out_type;
        out_type *svc(in_type*) {return nullptr;}
    };
    const emptyNode n;   
    pipe_t empty_pipe(n);

    return empty_pipe;
}

template<typename T1, typename T2>
const ff_Farm<typename T1::in_type, typename T2::out_type> optimize_static(ff_Farm<typename T1::in_type, typename T2::out_type>& farm) {

}
#endif

    
} // namespace ff 
#endif /* FF_OPTIMIZE_HPP */
