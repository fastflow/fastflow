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
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_wrappers.hpp>
#include <ff/distributed/ff_dreceiver.hpp>
#include <ff/distributed/ff_dsender.hpp>
#include <ff/distributed/ff_dadapters.hpp>

#include <numeric>

#ifdef DFF_MPI
#include <ff/distributed/ff_dreceiverMPI.hpp>
#include <ff/distributed/ff_dsenderMPI.hpp>
#endif


template<typename T>
T getBackAndPop(std::vector<T>& v){
    T b = v.back();
    v.pop_back();
    return b;
}

namespace ff{
class dGroup : public ff::ff_farm {

    static inline std::unordered_map<int, int> vector2UMap(const std::vector<int> v){
        std::unordered_map<int,int> output;
        for(size_t i = 0; i < v.size(); i++) output[v[i]] = i;
        return output;
    }

    static inline std::map<int, int> vector2Map(const std::vector<int> v){
        std::map<int,int> output;
        for(size_t i = 0; i < v.size(); i++) output[v[i]] = i;
        return output;
    }

    struct ForwarderNode : ff_node { 
        ForwarderNode(std::function<void(void*, dataBuffer&)> f){
            this->serializeF = f;
        }
        ForwarderNode(std::function<void*(dataBuffer&)> f){
            this->deserializeF = f;
        }
        void* svc(void* input){return input;}
    };

    static ff_node* buildWrapperIN(ff_node* n){
        if (n->isMultiOutput()) return new ff_comb(new WrapperIN(new ForwarderNode(n->deserializeF)), n, true, false);
        return new WrapperIN(n);
    }

    static ff_node* buildWrapperOUT(ff_node* n, int id, int outputChannels){
        if (n->isMultiInput()) return new ff_comb(n, new WrapperOUT(new ForwarderNode(n->serializeF), id, 1, true), false, true);
        return new WrapperOUT(n, id, outputChannels);
    }

public:
    dGroup(ff_IR& ir){

        int outputChannels = 0;
        if (ir.hasSender){
            ff::cout << "Size of routintable " << ir.routingTable.size() << std::endl;
            outputChannels = std::accumulate(ir.routingTable.begin(), ir.routingTable.end(), 0, [](const auto& s, const auto& f){return s+f.second.first.size();});
            ff::cout << "Outputchannels: " << outputChannels << std::endl;
        }
        if (ir.isVertical()){

            std::vector<int> reverseOutputIndexes(ir.hasLeftChildren() ? ir.outputL.rbegin() : ir.outputR.rbegin(), ir.hasLeftChildren() ? ir.outputL.rend() : ir.outputR.rend());
            for(ff_node* child: (ir.hasLeftChildren() ? ir.L : ir.R)){
                ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                
                // handle the case we have a pipe (or more nested) with just one sequential stage (not a combine)
                if (inputs.size() == 1 && outputs.size() == 1 && inputs[0] == outputs[0])
                    child = inputs[0];

               if (isSeq(child)){
                    if (ir.hasReceiver && ir.hasSender)
                        workers.push_back(new WrapperINOUT(child, getBackAndPop(reverseOutputIndexes)));
                    else if (ir.hasReceiver)
                        workers.push_back(buildWrapperIN(child));
                    else  workers.push_back(buildWrapperOUT(child, getBackAndPop(reverseOutputIndexes), outputChannels));

               } else {
                   if (ir.hasReceiver){
                        for(ff_node* input : inputs){
                            ff_node* inputParent = getBB(child, input);
                            if (inputParent) inputParent->change_node(input, buildWrapperIN(input), true); //cleanup?? removefromcleanuplist??
                        }
                   }

                   if (ir.hasSender){
                        for(ff_node* output : outputs){
                            ff_node* outputParent = getBB(child, output);
                            if (outputParent) outputParent->change_node(output, buildWrapperOUT(output, getBackAndPop(reverseOutputIndexes), outputChannels), true); // cleanup?? removefromcleanuplist??
                        }
                   }

                   workers.push_back(child);
               }
            }

            if (ir.hasReceiver)
                this->add_emitter(new ff_dreceiver(ir.listenEndpoint, ir.expectedEOS, vector2Map(ir.hasLeftChildren() ? ir.inputL : ir.inputR)));

            if (ir.hasSender)
                this->add_collector(new ff_dsender(ir.destinationEndpoints, ir.listenEndpoint.groupName), true);
        }
        else { // the group is horizontal!
            ff_a2a* innerA2A = new ff_a2a();
            
            std::vector<int> reverseLeftOutputIndexes(ir.outputL.rbegin(), ir.outputL.rend());

            std::unordered_map<int, int> localRightWorkers = vector2UMap(ir.inputR);
            std::vector<ff_node*> firstSet;
            for(ff_node* child : ir.L){
                if (isSeq(child))
                    if (ir.isSource){
                        ff_node* wrapped = new EmitterAdapter(child, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers);
                        wrapped->skipallpop(true);
                        firstSet.push_back(wrapped);
                    } else {
                        firstSet.push_back(new ff_comb(new WrapperIN(new ForwarderNode(child->getDeserializationFunction()), 1, true), new EmitterAdapter(child, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers), true, true));
                    }
                else {
                    
                    ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                    for(ff_node* input : inputs){
                        if (ir.isSource)
                            input->skipallpop(true);
                        else {
                            ff_node* inputParent = getBB(child, input);
                            if (inputParent) inputParent->change_node(input, buildWrapperIN(input), true); // cleanup??? remove_fromcleanuplist??
                        }
                    }
                    
                    ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                    for(ff_node* output : outputs){
                        ff_node* outputParent = getBB(child, output);
                        if (outputParent) outputParent->change_node(output,  new EmitterAdapter(output, ir.rightTotalInputs, getBackAndPop(reverseLeftOutputIndexes) , localRightWorkers) , true); // cleanup??? remove_fromcleanuplist??
                    }
                    firstSet.push_back(child); //ondemand?? cleanup??
                }
            }
            // add the Square Box Left, just if we have a receiver!
            if (ir.hasReceiver)
                firstSet.push_back(new SquareBoxLeft(localRightWorkers)); // ondemand??
            
            std::transform(firstSet.begin(), firstSet.end(), firstSet.begin(), [](ff_node* n) -> ff_node* {
                if (!n->isPipe())
					return new ff_Pipe(n);
				return n;
            });

            innerA2A->add_firstset(firstSet); // ondemand ??? clenaup??
            

            std::vector<int> reverseRightOutputIndexes(ir.outputR.rbegin(), ir.outputR.rend());
            std::vector<ff_node*> secondSet;
            for(ff_node* child : ir.R){
                if (isSeq(child))
                    secondSet.push_back(
                        (ir.isSink) ? (ff_node*)new CollectorAdapter(child, ir.outputL) 
                                    : (ff_node*)new ff_comb(new CollectorAdapter(child, ir.outputL), new WrapperOUT(new ForwarderNode(child->getSerializationFunction()), getBackAndPop(reverseRightOutputIndexes), 1, true), true, true)
                    );
                else {
                    ff::svector<ff_node*> inputs; child->get_in_nodes(inputs);
                    for(ff_node* input : inputs){
                        ff_node* inputParent = getBB(child, input);
                        if (inputParent) inputParent->change_node(input, new CollectorAdapter(input, ir.outputL), true); //cleanup?? remove_fromcleanuplist??
                    }

                    if (!ir.isSink){
                        ff::svector<ff_node*> outputs; child->get_out_nodes(outputs);
                        for(ff_node* output : outputs){
                            ff_node* outputParent = getBB(child, output);
                            if (outputParent) outputParent->change_node(output, buildWrapperOUT(output, getBackAndPop(reverseRightOutputIndexes), outputChannels), true); //cleanup?? removefromcleanuplist?
                        }
                    }

                    secondSet.push_back(child);
                }
            }
            
            // add the SQuareBox Right, iif there is a sender!
            if (ir.hasSender)
                secondSet.push_back(new SquareBoxRight);

            std::transform(secondSet.begin(), secondSet.end(), secondSet.begin(), [](ff_node* n) -> ff_node* {
                if (!n->isPipe())
					return new ff_Pipe(n);
				return n;
            });

            innerA2A->add_secondset<ff_node>(secondSet); // cleanup??
            workers.push_back(innerA2A);

            
            if (ir.hasReceiver)
                this->add_emitter(new ff_dreceiverH(ir.listenEndpoint, ir.expectedEOS, vector2Map(ir.inputL), ir.inputR, ir.otherGroupsFromSameParentBB));

            if (ir.hasSender)
                this->add_collector(new ff_dsenderH(ir.destinationEndpoints, ir.listenEndpoint.groupName, ir.otherGroupsFromSameParentBB) , true);
        }  

        if (this->getNWorkers() == 0){
            std::cerr << "The farm implementing the distributed group is empty! There might be an error! :(\n";
            abort();
        }
    }
         

};
}
#endif

