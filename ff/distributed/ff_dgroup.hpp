/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
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

/* Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */


#ifndef FF_DGROUP_H
#define FF_DGROUP_H

#include <ff/ff.hpp>
#include <ff/distributed/ff_ddefines.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_wrappers.hpp>
#include <ff/distributed/ff_dreceiverMTCL.hpp>
#include <ff/distributed/ff_dsenderMTCL.hpp>
#include <ff/distributed/ff_dadapters.hpp>

#include <numeric>

#ifdef DFF_MPI
#include <ff/distributed/ff_dreceiverMPI.hpp>
#include <ff/distributed/ff_dsenderMPI.hpp>
#endif



namespace ff{
class dGroup : public ff::ff_farm {
    const std::vector<ff_node*> emptyV = {};
    std::vector<ff_node*> tobeCleaned = {};

    struct ForwarderMiNode : ff_minode {
        void* svc(void* in) {return in;}
    };

    struct ForwarderMoNode : ff_monode {
        void* svc(void* in) {return in;}
    };

    struct ForwarderNode : ff_node { 
        ForwarderNode(bool (*f)(void*, message2_t* b),
					  void (*d)(void*)) {			
            this->serializeF = f;
			this->freetaskF  = d;
        }
        ForwarderNode(void* (*f)(message2_t*, bool&, ff_node*),
					  void* (*a)(char*, size_t)) {
			this->alloctaskF   = a;
            this->deserializeF = f;
        }

        ForwarderNode(ff_node* n){
            this->mioID = n->mioID;
            this->alloctaskF = n->alloctaskF;
            this->deserializeF = n->deserializeF;
            this->serializeF = n->serializeF;
            this->freetaskF = n->freetaskF;
        }

        void* svc(void* input){ return input;}
    };

    inline void _addWorker_(std::vector<ff_node*>& v, ff_node* n, bool cleanup = false){
        v.push_back(n);
        if (cleanup) tobeCleaned.push_back(n);
    }

    static inline ff_node* buildWrapperCollector(ff_node* n, IngressChannels_t& channels, const std::vector<ff_node*>& prevLocalWorkers){
        if (channels.empty()) return n;
        if (n->isMultiOutput())
            return new ff_comb(new CollectorAdapter2(new ForwarderNode(n), channels, prevLocalWorkers, true), n, true, false);
        return new CollectorAdapter2(n, channels, prevLocalWorkers);
        //return new ff_comb(new CollectorAdapter2(n, channels, prevLocalWorkers), new ForwarderMoNode,  true, true);
    }

    static inline ff_node* buildWrapperEmitter(ff_node* n, EgressChannels_t& channels, const std::vector<ff_node*>& nextLocalWorkers){
        if (channels.empty()) return n;
        if (n->isMultiInput()) 
            return new ff_comb(n, new EmitterAdapter2(new ForwarderNode(n), channels, nextLocalWorkers, false, true), false, true);
        return new EmitterAdapter2(n, channels, nextLocalWorkers);
    }

    static inline  ff_node* buildWrapperSeq(ff_node* n, IngressEgressChannels_t& channels, const std::vector<ff_node*>& prevLocalWorkers, const std::vector<ff_node*>& nextLocalWorkers){
        if (channels.first.empty()){
            return new EmitterAdapter2(n, channels.second, nextLocalWorkers, true);
        }
        if (channels.second.empty())
            return new ff_comb(new CollectorAdapter2(n, channels.first, prevLocalWorkers), new ForwarderMoNode, true, true);

        if (n->isMultiInput()){
            return new ff_comb(new CollectorAdapter2(n, channels.first, prevLocalWorkers), new EmitterAdapter2(new ForwarderNode(n), channels.second, nextLocalWorkers, false, true), true, true);
        }
        return new ff_comb(new CollectorAdapter2(new ForwarderNode(n), channels.first, prevLocalWorkers, true), new EmitterAdapter2(n, channels.second, nextLocalWorkers), true, true);
    }

public:

    dGroup(ff_IR_V2& ir) : ff_farm(){

        if (ir.bucketsDistribution.size() == 1){

            std::vector<ff_node*> wrappedWorkers;

            for(ff_node* n : ir.bucketsDistribution.front()){
                if (isSeq(n)) _addWorker_(wrappedWorkers, new WrapperINOUT(n, ir.channelsDictionary[n]), true);
                else if (n->isPipe() && reinterpret_cast<ff_pipeline*>(n)->cardinality() == 1) { // if i have just one worker in the pipeline i treat it like a sequential
                    ff_node* s = reinterpret_cast<ff_pipeline*>(n)->get_firststage();
                    _addWorker_(wrappedWorkers, new WrapperINOUT(s, ir.channelsDictionary[s]), true);
                }
                else {
                    if (!ir.ingressRemoteConnectionsGroupsName.empty()){ // if the receiver is present build the wrappers
                        ff::svector<ff_node*> inputNodes; n->get_in_nodes(inputNodes);
                        for (ff_node* in : inputNodes){
                            ff_node* inputParent = getBB(n, in);
                            if (inputParent) {
                                ff_node* wrapper = buildWrapperCollector(in, ir.channelsDictionary[in].first, emptyV);							
                                inputParent->change_node(in, wrapper, true, false); //cleanup?? removefromcleanuplist??
                            }  
                        }
                    }

                    if (!ir.destinationEndpoints.empty()){
                        ff::svector<ff_node*> outNodes; n->get_out_nodes(outNodes);
                        for (ff_node* out : outNodes){
                            ff_node* outParent = getBB(n, out);
                            if (outParent){
                                ff_node* wrapper = buildWrapperEmitter(out, ir.channelsDictionary[out].second, emptyV);
                                outParent->change_node(out, wrapper, true, false);
                            }
                        }
                    }

                    _addWorker_(wrappedWorkers, n);
                }
            }

            this->add_workers(wrappedWorkers);
            this->cleanup_workers(false);

        } else {
            // if i have multiple levels, i need to build the nested a2a
            ff_a2a* rootA2A = new ff_a2a;
            ff_a2a* current_A2A = rootA2A;
            for(size_t i = 0; i < ir.bucketsDistribution.size(); i++){
                std::vector<ff_node*> wrappedWorkers;
                const std::vector<ff_node*>& prevLocalWorkers = i > 0 ? ir.bucketsDistribution[i-1] : (ir.groupWrappedAround ? ir.bucketsDistribution.back() : emptyV);
                const std::vector<ff_node*>& nextLocalWorkers = i < (ir.bucketsDistribution.size() - 1) ? ir.bucketsDistribution[i+1] : (ir.groupWrappedAround ? ir.bucketsDistribution.front() : emptyV);
                for(ff_node* n : ir.bucketsDistribution[i])
                    if (isSeq(n)) 
                        _addWorker_(wrappedWorkers, buildWrapperSeq(n, ir.channelsDictionary[n], prevLocalWorkers, nextLocalWorkers), true);
                    else if (n->isPipe() && reinterpret_cast<ff_pipeline*>(n)->cardinality() == 1) { // if i have just one worker in the pipeline i treat it like a sequential
                        ff_node* s = reinterpret_cast<ff_pipeline*>(n)->get_firststage();
                        _addWorker_(wrappedWorkers, buildWrapperSeq(s, ir.channelsDictionary[s], prevLocalWorkers, nextLocalWorkers), true);
                    }
                    else {
                        if (!ir.ingressRemoteConnectionsGroupsName.empty()){ // if the receiver is present build the wrappers
                            ff::svector<ff_node*> inputNodes; n->get_in_nodes(inputNodes);
                            for (ff_node* in : inputNodes){
                                ff_node* inputParent = getBB(n, in);
                                if (inputParent) {
                                    ff_node* wrapper = buildWrapperCollector(in, ir.channelsDictionary[in].first, prevLocalWorkers);							
                                    inputParent->change_node(in, wrapper, true, false); //cleanup?? removefromcleanuplist??
                                }  
                            }
                        }

                        if (!ir.destinationEndpoints.empty()){
                            ff::svector<ff_node*> outNodes; n->get_out_nodes(outNodes);
                            for (ff_node* out : outNodes){
                                ff_node* outParent = getBB(n, out);
                                if (outParent){
                                    ff_node* wrapper = buildWrapperEmitter(out, ir.channelsDictionary[out].second, nextLocalWorkers);
                                    outParent->change_node(out, wrapper, true, false);
                                }
                            }
                        }


                        wrappedWorkers.push_back(n);
                    }
                
                // add also the squarebox to wrapped workers
                _addWorker_(wrappedWorkers, new ff_comb(new SquareBoxInputAdapter, new SquareBox(nextLocalWorkers, ir.channelsDictionary), true, true), true); // add combine to have either multi input and multioutput
        
                /*if (i == 0)
                    current_A2A->add_firstset(wrappedWorkers, 0, true);
                else*/ if (i == (ir.bucketsDistribution.size()-1)){
                    current_A2A->add_secondset(wrappedWorkers, false);
                } else {
                    current_A2A->add_firstset(wrappedWorkers, 0, false);
                    
                    if (i+2 == ir.bucketsDistribution.size()) continue;
                    
                    ff_a2a* nextLevelA2A = new ff_a2a;

                    // i need to wrap the nested a2a in a pipe in order to add to the external a2a
                    ff_pipeline* p = new ff_pipeline;
                    p->add_stage(nextLevelA2A, true);

                    current_A2A->add_secondset<ff_pipeline>({p}, true);
                    current_A2A = nextLevelA2A;
                }
            }
            // check if there is a feedback internally

            if (ir.groupWrappedAround) rootA2A->wrap_around();

            this->add_workers({rootA2A});
            this->cleanup_workers(true);
        }

        if (!ir.destinationEndpoints.empty()){
            this->add_collector(new ff_dsenderMTCL2(ir.destinationEndpoints, ir.channelsDictionary, ir.batchSize, ir.batchByteSize, ir.messageOTF), true); // TO DO: check the signature of the dsender constructor
        }

        if (!ir.ingressRemoteConnectionsGroupsName.empty()){
            this->add_emitter(new ff_comb(new ff_dreceiverMTCL2(ir.listeningEndpoint, ir.ingressRemoteConnectionsGroupsName.size(), ir.bucketsDistribution.size() == 1), new SquareBox(ir.bucketsDistribution.front(), ir.channelsDictionary, ir.bucketsDistribution.size() > 1), true, true));
            this->cleanup_emitter(true);
        }

        ff::termination_counter = this->cardinality() - 1 /*receiver*/ - (ir.bucketsDistribution.size() > 1 ? ir.bucketsDistribution.size() + 1 /*Sender just for multi level groups*/ : 0);
    }

    ~dGroup(){
        for(ff_node* n : tobeCleaned) delete n;
    }

};
} // namespace
#endif

