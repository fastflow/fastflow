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

#include <list>
#include <numeric>

#ifdef DFF_MPI
#include <ff/distributed/ff_dreceiverMPI.hpp>
#include <ff/distributed/ff_dsenderMPI.hpp>
#endif



namespace ff{
class dGroup : public ff::ff_farm {
    const std::vector<ff_node*> emptyV = {};
    std::list<std::vector<ff_node*>> localNodesStorage = {};
    std::vector<ff_node*> tobeCleaned = {};

    struct ForwarderMiNode : ff_minode {
        void* svc(void* in) {return in;}
    };

    struct ForwarderMoNode : ff_monode {
        void* svc(void* in) {return in;}
    };

    struct ForwarderNode : ff_node { 
        ForwarderNode(bool (*f)(void*, message2_t*),
					  void (*d)(void*), 
                      void(*freeBlob)(char*,size_t)
                      ) {			
            this->serializeF = f;
			this->freetaskF  = d;
            this->freeBlob = freeBlob;
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
            this->freeBlob = n->freeBlob;
        }

        void* svc(void* input){ return input;}
    };

    inline void _addWorker_(std::vector<ff_node*>& v, ff_node* n, bool cleanup = false){
        v.push_back(n);
        if (cleanup) tobeCleaned.push_back(n);
    }

    const std::vector<ff_node*>& keepLocalNodes(const ff::svector<ff_node*>& nodes) {
        localNodesStorage.emplace_back(nodes.begin(), nodes.end());
        return localNodesStorage.back();
    }

    static inline void printIndent(size_t indent) {
        for(size_t i = 0; i < indent; ++i) ff::cout << "  ";
    }

    static inline const char* nodeKind(ff_node* n) {
        if (n->isAll2All()) return "a2a";
        if (n->isPipe()) return "pipe";
        if (n->isFarm()) return "farm";
        if (n->isComp()) return "comp";
        if (n->isMultiInput() && n->isMultiOutput()) return "mi+mo";
        if (n->isMultiInput()) return "mi";
        if (n->isMultiOutput()) return "mo";
        return "seq";
    }

    // Recursively dump the concrete nodes inserted in the runtime group so we
    // can compare them with the higher-level IR buckets when debugging cuts.
    static inline void printNode(ff_node* n, size_t indent) {
        printIndent(indent);
        ff::cout << "- " << nodeKind(n)
                 << " node@" << n
                 << " id=" << n->mioID_str << "[" << n->mioID << "]"
                 << " mi=" << n->isMultiInput()
                 << " mo=" << n->isMultiOutput()
                 << " comp=" << n->isComp()
                 << "\n";

        if (n->isPipe()) {
            const auto& stages = reinterpret_cast<ff_pipeline*>(n)->getStages();
            for(size_t i = 0; i < stages.size(); ++i) {
                printIndent(indent + 1);
                ff::cout << "stage " << i << ":\n";
                printNode(stages[i], indent + 2);
            }
            return;
        }

        if (n->isAll2All()) {
            auto* a2a = reinterpret_cast<ff_a2a*>(n);
            const auto& firstSet = a2a->getFirstSet();
            const auto& secondSet = a2a->getSecondSet();

            printIndent(indent + 1);
            ff::cout << "first-set (" << firstSet.size() << ")\n";
            for(ff_node* worker : firstSet)
                printNode(worker, indent + 2);

            printIndent(indent + 1);
            ff::cout << "second-set (" << secondSet.size() << ")\n";
            for(ff_node* worker : secondSet)
                printNode(worker, indent + 2);
            return;
        }

        if (n->isFarm()) {
            auto* farm = reinterpret_cast<ff_farm*>(n);
            if (ff_node* emitter = farm->getEmitter()) {
                printIndent(indent + 1);
                ff::cout << "emitter:\n";
                printNode(emitter, indent + 2);
            }

            const auto& workers = farm->getWorkers();
            printIndent(indent + 1);
            ff::cout << "workers (" << workers.size() << ")\n";
            for(ff_node* worker : workers)
                printNode(worker, indent + 2);

            if (ff_node* collector = farm->getCollector()) {
                printIndent(indent + 1);
                ff::cout << "collector:\n";
                printNode(collector, indent + 2);
            }
        }
    }

    static inline ff_node* buildWrapperCollector(ff_node* n, IngressChannels_t& channels, const std::vector<ff_node*>& prevLocalWorkers, bool prevLevelSquareBox){
        if (channels.empty()) return n;
        if (n->isMultiOutput())
            return new ff_comb(new CollectorAdapter2(new ForwarderNode(n), channels, prevLocalWorkers, prevLevelSquareBox, true), n, true, false);
        return new CollectorAdapter2(n, channels, prevLocalWorkers, prevLevelSquareBox);
        //return new ff_comb(new CollectorAdapter2(n, channels, prevLocalWorkers), new ForwarderMoNode,  true, true);
    }

    static inline ff_node* buildWrapperEmitter(ff_node* n, EgressChannels_t& channels, const std::vector<ff_node*>& nextLocalWorkers, bool nextLevelSquareBox){
        if (channels.empty()) return n;
        if (n->isMultiInput()) 
            return new ff_comb(n, new EmitterAdapter2(new ForwarderNode(n), channels, nextLocalWorkers, nextLevelSquareBox, false, true), false, true);
        return new EmitterAdapter2(n, channels, nextLocalWorkers, nextLevelSquareBox);
    }

    static inline  ff_node* buildWrapperSeq(ff_node* n, IngressEgressChannels_t& channels, const std::vector<ff_node*>& prevLocalWorkers, const std::vector<ff_node*>& nextLocalWorkers, bool prevLevelSquareBox, bool nextLevelSquareBox){
        if (channels.first.empty()){
            return new EmitterAdapter2(n, channels.second, nextLocalWorkers, nextLevelSquareBox, true);
        }
        if (channels.second.empty())
            return /*new ff_comb(*/new CollectorAdapter2(n, channels.first, prevLocalWorkers, prevLevelSquareBox)/*, new ForwarderMoNode, true, true)*/;

        if (n->isMultiInput()){
            return new ff_comb(new CollectorAdapter2(n, channels.first, prevLocalWorkers, prevLevelSquareBox), new EmitterAdapter2(new ForwarderNode(n), channels.second, nextLocalWorkers, nextLevelSquareBox,  false, true), true, true);
        }
        return new ff_comb(new CollectorAdapter2(new ForwarderNode(n), channels.first, prevLocalWorkers, prevLevelSquareBox, true), new EmitterAdapter2(n, channels.second, nextLocalWorkers, nextLevelSquareBox), true, true);
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
                    ff::svector<ff_node*> inputNodes;
                    ff::svector<ff_node*> outputNodes;
                    n->get_in_nodes(inputNodes);
                    n->get_out_nodes(outputNodes);

                    // In a single-level wrapped group, composite skeleton
                    // boundary nodes may still communicate locally through
                    // feedback channels. Keep these vectors alive because the
                    // adapters store references to them.
                    const std::vector<ff_node*>& prevLocalWorkers =
                        ir.groupWrappedAround ? keepLocalNodes(outputNodes) : emptyV;
                    const std::vector<ff_node*>& nextLocalWorkers =
                        ir.groupWrappedAround ? keepLocalNodes(inputNodes) : emptyV;

                    if (!ir.ingressRemoteConnectionsGroupsName.empty()){ // if the receiver is present build the wrappers
                        for (ff_node* in : inputNodes){
                            ff_node* inputParent = getBB(n, in);
                            if (inputParent) {
                                ff_node* wrapper = buildWrapperCollector(in, ir.channelsDictionary[in].first, prevLocalWorkers, true);
                                inputParent->change_node(in, wrapper, true, false); //cleanup?? removefromcleanuplist??
                            }  
                        }
                    }

                    if (!ir.destinationEndpoints.empty()){
                        for (ff_node* out : outputNodes){
                            ff_node* outParent = getBB(n, out);
                            if (outParent){
                                ff_node* wrapper = buildWrapperEmitter(out, ir.channelsDictionary[out].second, nextLocalWorkers, true);
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
                const bool prevLevelSquareBox = (i-1 == 0 && !ir.ingressRemoteConnectionsGroupsName.empty()) || (i-1 > 0);
                const bool nextLevelSquareBox = (i+2 == ir.bucketsDistribution.size() && !ir.destinationEndpoints.empty()) || (i+1 == ir.bucketsDistribution.size() && !ir.destinationEndpoints.empty()) || (i+2 < ir.bucketsDistribution.size());
                const std::vector<ff_node*>& prevLocalWorkers = i > 0 ? ir.bucketsDistribution[i-1] : (ir.groupWrappedAround ? ir.bucketsDistribution.back() : emptyV);
                const std::vector<ff_node*>& nextLocalWorkers = i < (ir.bucketsDistribution.size() - 1) ? ir.bucketsDistribution[i+1] : (ir.groupWrappedAround ? ir.bucketsDistribution.front() : emptyV);
                for(ff_node* n : ir.bucketsDistribution[i])
                    if (isSeq(n)) 
                        _addWorker_(wrappedWorkers, buildWrapperSeq(n, ir.channelsDictionary[n], prevLocalWorkers, nextLocalWorkers, prevLevelSquareBox, nextLevelSquareBox), true);
                    else if (n->isPipe() && reinterpret_cast<ff_pipeline*>(n)->cardinality() == 1) { // if i have just one worker in the pipeline i treat it like a sequential
                        ff_node* s = reinterpret_cast<ff_pipeline*>(n)->get_firststage();
                        _addWorker_(wrappedWorkers, buildWrapperSeq(s, ir.channelsDictionary[s], prevLocalWorkers, nextLocalWorkers, prevLevelSquareBox, nextLevelSquareBox), true);
                    }
                    else {
                        if (!ir.ingressRemoteConnectionsGroupsName.empty()){ // if the receiver is present build the wrappers
                            ff::svector<ff_node*> inputNodes; n->get_in_nodes(inputNodes);
                            for (ff_node* in : inputNodes){
                                ff_node* inputParent = getBB(n, in);
                                if (inputParent) {
                                    ff_node* wrapper = buildWrapperCollector(in, ir.channelsDictionary[in].first, prevLocalWorkers, prevLevelSquareBox);							
                                    inputParent->change_node(in, wrapper, true, false); //cleanup?? removefromcleanuplist??
                                }  
                            }
                        }

                        if (!ir.destinationEndpoints.empty() || !nextLocalWorkers.empty()){
                            ff::svector<ff_node*> outNodes; n->get_out_nodes(outNodes);
                            for (ff_node* out : outNodes){
                                ff_node* outParent = getBB(n, out);
                                if (outParent){
                                    // Local edges between levels need the emitter
                                    // adapter too, otherwise downstream collectors
                                    // receive only real EOS and miss logical EOS.
                                    ff_node* wrapper = buildWrapperEmitter(out, ir.channelsDictionary[out].second, nextLocalWorkers, nextLevelSquareBox);
                                    outParent->change_node(out, wrapper, true, false);
                                }
                            }
                        }


                        wrappedWorkers.push_back(n);
                    }
                
                // add also the squarebox to wrapped workers
                if ((i == 0 && !ir.ingressRemoteConnectionsGroupsName.empty()) || (i == (ir.bucketsDistribution.size()-1) && !ir.destinationEndpoints.empty()) || (i != 0 && i != (ir.bucketsDistribution.size()-1))){
                    if (wrappedWorkers.front()->isMultiOutput())
                        _addWorker_(wrappedWorkers, new ff_comb(new SquareBoxInputAdapter, new SquareBox(nextLocalWorkers, ir.channelsDictionary, nextLevelSquareBox), true, true), true); // add combine to have either multi input and multioutput
                    else
                        _addWorker_(wrappedWorkers, new SquareBoxInputAdapter, true); // if the workers are not multioutput i can just add the square box as input adapter
                }
                
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

        ff::termination_counter = this->cardinality() - 1 /*receiver*/ - (ir.bucketsDistribution.size() > 1 ? ir.bucketsDistribution.size() + (-ir.destinationEndpoints.empty()) + (!ir.destinationEndpoints.empty()) /*Sender just for multi level groups*/ : 0);
        ff::cout << "Setting termination counter to: " << ff::termination_counter << std::endl;
    }

    ~dGroup(){
        for(ff_node* n : tobeCleaned) delete n;
    }

    void print(const std::string& groupName) const {
        ff::cout << "******** BEGIN BUILT GROUP " << groupName << " ********\n";

        if (ff_node* emitter = this->getEmitter()) {
            ff::cout << "Emitter:\n";
            printNode(emitter, 1);
        }

        ff::cout << "Workers (" << this->getWorkers().size() << "):\n";
        for(ff_node* worker : this->getWorkers())
            printNode(worker, 1);

        if (ff_node* collector = this->getCollector()) {
            ff::cout << "Collector:\n";
            printNode(collector, 1);
        }

        ff::cout << "********* END BUILT GROUP " << groupName << " *********\n";
    }

};
} // namespace
#endif
