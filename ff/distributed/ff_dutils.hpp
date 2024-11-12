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
 *
 * Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FFDUTILS_HPP
#define FFDUTILS_HPP
#include <set>
#include "ff/pipeline.hpp"

namespace ff {
enum NodeIOTypes { IN, OUT, INOUT};
enum SetEnum {L, R};

/**
 * Check the coverage of the first level building blocks of the main pipe, which is the building block needed to create a distributed FF program.
 * Specifically this method, given the main pipe and and the set of buildingblock from which the user created groups, check if all stages of the pipe belong to a group.
 **/
static inline bool checkCoverageFirstLevel(ff_pipeline* mainPipe, const std::set<ff_node*>& groupBuildingBlock) {
    ff::svector<ff_node*> stages = mainPipe->getStages();
    for(size_t i = 0; i< stages.size(); i++)
        if (!groupBuildingBlock.count(stages[i])){
            std::cerr << "Stage #" << i << " was not annotated in any group! Aborting!\n";
            abort();
        }
    
    return true;
}

/**
 * Helper function to detect sequential node from a bare node pointer.
 **/
static inline bool isSeq(const ff_node* n){return (!n->isAll2All() && !n->isComp() && !n->isFarm() && !n->isOFarm() && !n->isPipe());}

/**
 * Return the children builiding block of the given building block. We implemented only a2a and pipeline since, groups can be created just only from this two building block.
 **/
static inline std::set<std::pair<ff_node*, SetEnum>> getChildBB(ff_node* parent){
    std::set<std::pair<ff_node*, SetEnum>> out;
    if (parent->isAll2All()){
        for (ff_node* bb : reinterpret_cast<ff_a2a*>(parent)->getFirstSet())
            out.emplace(bb, SetEnum::L);
        
        for(ff_node* bb : reinterpret_cast<ff_a2a*>(parent)->getSecondSet())
            out.emplace(bb, SetEnum::R);
    }

    if (parent->isPipe())
        for(ff_node* bb : reinterpret_cast<ff_pipeline*>(parent)->getStages())
            out.emplace(bb, SetEnum::L); // for pipelines the default List is L (left)
    
    return out;
}

static inline bool isSource(const ff_node* n, const ff_pipeline* p){
    return p->getStages().front() == n;
}

static inline bool isSink(const ff_node* n, const ff_pipeline* p){
    return p->getStages().back() == n;
}

static inline ff_node* getPreviousStage(ff_pipeline* p, ff_node* s){
    ff::svector<ff_node*> stages = p->getStages();
    for(size_t i = 1; i < stages.size(); i++)
        if (stages[i] == s) return stages[--i];
    
    return nullptr;
}

static inline ff_node* getNextStage(ff_pipeline* p, ff_node* s){
    ff::svector<ff_node*> stages = p->getStages();
    for(size_t i = 0; i < stages.size() - 1; i++)
        if(stages[i] == s) return stages[++i];
    
    return nullptr;
}




/*
* FUNCTIONS ADDED IN THE NEW VERSION OF DISTRIBUTED RTS
*/


// Utility for the next two functions to merge pairs of svectors
inline std::pair<ff::svector<ff_node*>,ff::svector<ff_node*>> operator+=(std::pair<ff::svector<ff_node*>,ff::svector<ff_node*>>& a, const std::pair<ff::svector<ff_node*>,ff::svector<ff_node*>>& b){
    a.first += b.first;
    a.second += b.second;
    return a;
}

inline std::tuple<ff::svector<ff_node*>,ff::svector<ff_node*>, int> operator+=(std::tuple<ff::svector<ff_node*>,ff::svector<ff_node*>, int>& a, const std::tuple<ff::svector<ff_node*>,ff::svector<ff_node*>, int>& b){
    std::get<0>(a) += std::get<0>(b);
    std::get<1>(a) += std::get<1>(b);
    std::get<2>(a) = std::max(std::get<2>(a), std::get<2>(b));
    return a;
}


std::tuple<ff::svector<ff_node*>,ff::svector<ff_node*>, int> getConsumers(ff_node* n, ff_node* root){
    std::tuple<ff::svector<ff_node*>, ff::svector<ff_node*>, int> out = {{}, {}, 0};
    if (n == root) return out; // base case

    ff_node* parent = getBB(root, n);
    if (!parent) return out;
    if (parent->isPipe()){
        ff_pipeline* parentPipe = reinterpret_cast<ff_pipeline*>(parent);
        ff_node* next_stage = parentPipe->get_nextstage(n);
        if (next_stage) {
            next_stage->get_in_nodes(std::get<0>(out));
            return out;
        } else {
            if (parentPipe->isset_wraparound())
                parentPipe->get_in_nodes(std::get<1>(out)); // NB: maybe we should use get_in_nodes_feedback

            return out += getConsumers(parent, root); // merge of vector containing feedback consumers + forward consumers
        }
    }

    if (parent->isAll2All()){
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(parent);
        if (isFromFirstSet(n, a2a)){
            for(auto* c : a2a->getSecondSet())
                c->get_in_nodes(std::get<0>(out));
            if (a2a->ondemand_buffer() > 0) std::get<2>(out) = a2a->ondemand_buffer();
            return out;
        } else {
            if (a2a->isset_wraparound())
                a2a->get_in_nodes(std::get<1>(out));
            return out += getConsumers(parent, root);
        }
    }

    if (parent->isComp())
        return getConsumers(parent, root);

    std::cerr << "If you see this message there is for sure an ERROR!!!\n";
    return out;
}

std::tuple<ff::svector<ff_node*>,ff::svector<ff_node*>, int> getFeeders(ff_node* n, ff_node* root){
    std::tuple<ff::svector<ff_node*>, ff::svector<ff_node*>, int> out = {{}, {}, 0};
    if (n == root) return out; // base case

    ff_node* parent = getBB(root, n);
    if (!parent) return out;
    if (parent->isPipe()){
        ff_pipeline* parentPipe = reinterpret_cast<ff_pipeline*>(parent);
        ff_node* prev_stage = parentPipe->get_prevstage(n);
        if (prev_stage) {
            prev_stage->get_out_nodes(std::get<0>(out));
            return out;
        } else {
            if (parentPipe->isset_wraparound())
                parentPipe->get_out_nodes(std::get<1>(out)); // NB: maybe we should use get_out_nodes_feedback
            return out += getFeeders(parent, root); // merge of vector containing feedback feeders + forward feeders
        }
    }

    if (parent->isAll2All()){
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(parent);
        if (isFromSecondSet(n, a2a)){
            for(auto* c : a2a->getFirstSet())
                c->get_out_nodes(std::get<0>(out));
            if (a2a->ondemand_buffer() > 0) std::get<2>(out) = a2a->ondemand_buffer();
            return out;
        } else {
            if (a2a->isset_wraparound())
                a2a->get_out_nodes(std::get<1>(out)); // NB: maybe we should use get_out_nodes_feedback
            return out += getFeeders(parent, root); // merge of vector containing feedback feeders + forward feeders
        }
    }

    if (parent->isComp())
        return getFeeders(parent, root);

    std::cerr << "If you see this message there is for sure an ERROR!!!\n";
    return out;
}

// functions for handling the labels to perform the sorting operations on groups

void setIDs(ff_node* root, std::string prefix = ""){
    static int id_counter = 0;
    // set the ids to all building blocks
    
    if (root->isComp()){
        ff_comb* comp = reinterpret_cast<ff_comb*>(root);
        auto* first  = comp->getFirst();
        auto* last = comp->getLast();

        root->mioID = first->mioID = last->mioID = id_counter++;
        root->mioID_str = first->mioID_str = last->mioID_str = prefix;
        return;
    }

    root->mioID = id_counter++;
    root->mioID_str = prefix;

    if (root->isPipe()){
        int i = 0;
        for(auto& stage : reinterpret_cast<ff_pipeline*>(root)->getStages())
            setIDs(stage, prefix+"P"+std::to_string(i++));
    } else
    if (root->isAll2All()){
        auto& leftV = reinterpret_cast<ff_a2a*>(root)->getFirstSet();
        auto& rightV = reinterpret_cast<ff_a2a*>(root)->getSecondSet();

        int i = 0, j = 0;
        for (auto& w : leftV) setIDs(w, prefix+"S"+std::to_string(i++));
        for (auto& w : rightV) setIDs(w, prefix+"D"+std::to_string(j++));
    }
}

int parseNumericValue(const std::string& str, size_t& i){
    std::string numStr;
    while (i < str.size() && isdigit(str[i]))
        numStr += str[i++];
    return std::stoi(numStr);
}

int compareLabels(const std::string& id1, const std::string& id2){
    size_t i1 = 0, i2 = 0;
    while(i1 < id1.size() && i2 < id2.size()){
        char k1 = id1[i1++];
        char k2 = id2[i2++];
        int v1 = parseNumericValue(id1, i1);
        int v2 = parseNumericValue(id2, i2);
        
        if (k1 == k2 && k1 == 'P' && v1 != v2)
            return (v1-v2);

        if (k1 == 'S' && k2 == 'D') return -1;
        if (k1 == 'D' && k2 == 'S') return 1;
    }
    //std::cout <<  "Errore! S1: " << id1 << " - S2: " << id2 << std::endl;
    return 0;
}

inline int compare_nodes(const ff_node* n1, const ff_node* n2){
    return compareLabels(n1->mioID_str, n2->mioID_str);
}

void sortBasedOnLabel(std::vector<ff_node*>& arr){ 
    for (size_t i = 0; i < arr.size()-1; i++) 
        // Last i elements are already in place
        for (size_t j = 0; j < arr.size() - i - 1; j++) 
            if (compare_nodes(arr[j],arr[j + 1]) > 0) 
                std::swap(arr[j], arr[j + 1]); 
}

std::vector<std::vector<ff_node*>> build_A2A_structure(const std::vector<ff_node*>& v){
    std::vector<std::vector<ff_node*>> out;
    out.reserve(v.size());
    size_t currentBucket = 0;
    out.push_back(std::vector<ff_node*>({v.front()}));

    for(size_t i = 1; i < v.size(); i++)
        if (compare_nodes(out[currentBucket].back(), v[i]) == 0) out[currentBucket].push_back(v[i]);
        else {
            ++currentBucket;
            out.push_back(std::vector<ff_node*>({v[i]}));
        }
    out.shrink_to_fit();
    return out;
}

void custom_get_in_nodes(ff_node* n, ff::svector<int>& w){
    if (!n) return;

    if (n->isPipe())
        return custom_get_in_nodes(reinterpret_cast<ff_pipeline*>(n)->get_firststage(), w);

    if (n->isComp())
        return custom_get_in_nodes(reinterpret_cast<ff_comb*>(n)->getFirst(), w);

    if (n->isFarm())
        return custom_get_in_nodes(reinterpret_cast<ff_farm*>(n)->getEmitter(), w);

    if (n->isAll2All()){
        for (ff_node* nn : reinterpret_cast<ff_a2a*>(n)->getFirstSet())
            custom_get_in_nodes(nn, w);
        return;
    }

    w.push_back(n->mioID);
}

void custom_get_out_nodes(ff_node* n, ff::svector<int>& w){
    if (!n) return;

    if (n->isPipe())
        return custom_get_out_nodes(reinterpret_cast<ff_pipeline*>(n)->get_firststage(), w);

    if (n->isComp())
        return custom_get_out_nodes(reinterpret_cast<ff_comb*>(n)->getLast(), w);

    if (n->isFarm()){
        ff_farm* f = reinterpret_cast<ff_farm*>(n);
        if (f->hasCollector())
            return custom_get_out_nodes(reinterpret_cast<ff_farm*>(n)->getCollector(), w);
        for(ff_node* worker : f->getWorkers())
            custom_get_out_nodes(worker, w);
        return;
    }

    if (n->isAll2All()){
        for (ff_node* nn : reinterpret_cast<ff_a2a*>(n)->getSecondSet())
            custom_get_out_nodes(nn, w);
        return;
    }

    w.push_back(n->mioID);
}


} // namespace
#endif