#ifndef FF_DGROUP_H
#define FF_DGROUP_H

#include <ff/ff.hpp>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <map>
#include <exception>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_wrappers.hpp>
#include <ff/distributed/ff_dreceiver.hpp>
#include <ff/distributed/ff_dsender.hpp>
#include <ff/distributed/ff_dgroups.hpp>

#include <cereal/details/traits.hpp> // used for operators constexpr evaulations


namespace ff{

class dGroup;

enum IOTypes { IN, OUT };

template <IOTypes>
class MySet {
private:
    dGroup* group;

    struct ForwarderNode : ff_node{ 
        void* svc(void* input){return input;}
    };
public:
    MySet(dGroup* group): group(group){ }

    template <typename Tin, typename Tout>
    MySet& operator<<(ff_node_t<Tin, Tout>*) {}

    template <typename Tin, typename Tout>
    MySet& operator<<(ff_minode_t<Tin, Tout>*) {}

    template <typename Tin, typename Tout>
    MySet& operator<<(ff_monode_t<Tin, Tout>*) {}

    template <typename Tin, typename Tout>
    MySet& operator<<=(ff_node_t<Tin, Tout>*) {}

    template <typename Tin, typename Tout, typename Function>
    MySet& operator<<=(std::pair<ff_node_t<Tin, Tout>*, Function>);

    template <typename Tin, typename Tout>
    MySet& operator<<=(ff_minode_t<Tin, Tout>*);

    template <typename Tin, typename Tout, typename Function>
    MySet& operator<<=(std::pair<ff_minode_t<Tin, Tout>*, Function>);

    template <typename Tin, typename Tout>
    MySet& operator<<=(ff_monode_t<Tin, Tout>*);

    template <typename Tin, typename Tout, typename Function>
    MySet& operator<<=(std::pair<ff_monode_t<Tin, Tout>*, Function>);


    bool check_inout(ff_node* node);
};

class dGroups;

class dGroup : public ff_farm {

    friend class MySet<IN>;
    friend class MySet<OUT>;
private:
    ff_node * parentStructure;

    ff_endpoint endpoint;
    std::vector<ff_endpoint> destinations;
    int expectedInputConnections;

    /**
     * Key: reference to original node
     * Value: pair of [reference to wrapper, serialization_required] 
     **/
    std::map<ff_node*, std::pair<ff_node*, bool>> in_, out_;
    std::map<ff_node*, ff_node*> inout_;

    bool isSource(){return in_.empty() && inout_.empty();}
    bool isSink(){return out_.empty() && inout_.empty();}

    static bool isIncludedIn(const ff::svector<ff_node*>& firstSet, std::vector<ff_node*>& secondSet){
        for (const ff_node* n : firstSet)
            if (std::find(secondSet.begin(), secondSet.end(), n) == secondSet.end())
                return false;
        return true;
    }

    bool replaceWrapper(const ff::svector<ff_node*>& list, std::map<ff_node*, std::pair<ff_node*, bool>>& wrappers_){
        for (ff_node* node : list){
            ff_node* parentBB = getBB(this->parentStructure, node);
            if (parentBB != nullptr){
                
                ff_node* wrapped = wrappers_[node].first;
                
                if (parentBB->isPipe()){
                    reinterpret_cast<ff_pipeline*>(parentBB)->change_node(node, wrapped, true, true);
                    continue;
                }

                if (parentBB->isAll2All()){
                    reinterpret_cast<ff_a2a*>(parentBB)->change_node(node, wrapped, true, true);
                    continue;
                }

                if (parentBB->isComp()){
                    reinterpret_cast<ff_comb*>(parentBB)->change_node(node, wrapped, true, true);
                    continue;
                }

                return false;
            }
            return false;
        }
        return true;
    }

    ff_node* getOriginal(ff_node* wrapper){
        auto resultI = std::find_if(this->in_.begin(), this->in_.end(), [&](const std::pair<ff_node*, std::pair<ff_node*, bool>> &pair){return pair.second.first == wrapper;});
        if (resultI != this->in_.end()) return resultI->first;
        auto resultII = std::find_if(this->inout_.begin(), this->inout_.end(), [&](const std::pair<ff_node*, ff_node*> &pair){return pair.second == wrapper;});
        if (resultII != this->inout_.end()) return resultII->first;

        return nullptr;
    }

    static inline bool isSeq(ff_node* n){return (!n->isAll2All() && !n->isComp() && !n->isFarm() && !n->isOFarm() && !n->isPipe());}

    bool processBB(ff_node* bb, std::vector<ff_node*> in_C, std::vector<ff_node*> out_C){
        if (isSeq(bb)){
            ff::svector<ff_node*> svectBB(1); svectBB.push_back(bb);
            if (isSource() && this->out_.find(bb) != this->out_.end() && replaceWrapper(svectBB, this->out_)){
                this->add_workers({this->out_[bb].first});
                return true;
            }

            if (isSink() && this->in_.find(bb) != this->in_.end() && replaceWrapper(svectBB, this->in_)){
                this->add_workers({this->in_[bb].first});
                return true;
            }

            return false;
        }
        
        ff::svector<ff_node*> in_nodes, out_nodes;
        bb->get_in_nodes(in_nodes);
        
        if (!isSource() && !isIncludedIn(in_nodes, in_C))
                return false;

        bb->get_out_nodes(out_nodes);
        
        if (!isSink() && !isIncludedIn(out_nodes, out_C))
                return false;
        
        if ((isSource() || replaceWrapper(in_nodes, this->in_)) && (isSink() || replaceWrapper(out_nodes, this->out_))){
            this->add_workers({bb}); // here the bb is already modified with the wrapper
            return true;
        }

        return false;
    }

    static bool isStageOf(ff_node* n, ff_pipeline* p){
        for (const ff_node* s : p->getStages())
            if (s == n) return true;

        return false;
    }

    static int getInputIndexOfNode(ff_node* bb, ff_node* wrapper, ff_node* original){
        if (bb->isAll2All()){
            ff_a2a* a2a = (ff_a2a*) bb;
            int index = 0;
            for (ff_node* n : a2a->getFirstSet()){
                ff::svector<ff_node*> inputs; n->get_in_nodes(inputs);
                for (const ff_node* input : inputs){
                    if (input == wrapper || input == original)
                        return index;
                    index++;
                }
            }

            index = 0;
            for (ff_node* n : a2a->getSecondSet()){
                ff::svector<ff_node*> inputs; n->get_in_nodes(inputs);
                for (ff_node* input : inputs)
                    if (input == wrapper || input == original) 
                        return index; 
                    else index++;
            }
        }

        int index = 0;
        ff::svector<ff_node*> inputs; bb->get_in_nodes(inputs);
        for (ff_node* input : inputs)
            if (input == wrapper || input == original) 
                return index; 
            else index++;

        return 0;
    }

    std::map<int, int> buildRoutingTable(ff_node* level1BB){
        std::map<int, int> routingTable;
        int localIndex = 0;
        for (ff_node* inputBB : this->getWorkers()){
            ff::svector<ff_node*> inputs; inputBB->get_in_nodes(inputs);
            for (ff_node* input : inputs){
                routingTable[getInputIndexOfNode(level1BB, input, getOriginal(input))] = localIndex;
                localIndex++;
            }
                //routingTable[getInputIndexOfNode(level1BB, reinterpret_cast<Wrapper*>(input)->getOriginal())] = localIndex++;
        }
        return routingTable;
    }



    int buildFarm(ff_pipeline* basePipe = nullptr){ // chimato dalla run & wait della main pipe 

        // find the 1 level builiding block which containes the group (level 1 BB means a BB whoch is a stage in the main piepline)
        ff_node* level1BB = this->parentStructure;
        while(!isStageOf(level1BB, basePipe)){
            level1BB = getBB(basePipe, level1BB);
            if (!level1BB || level1BB == basePipe) throw FF_Exception("A group were created from a builiding block not included in the Main Pipe! :(");
        }

        
        std::vector<ff_node*> in_C, out_C;
        for (const auto& pair : this->in_) in_C.push_back(pair.first);
        for (const auto& pair : this->out_) out_C.push_back(pair.first);


        if (this->parentStructure->isPipe())
           processBB(this->parentStructure, in_C, out_C);


        if (this->parentStructure->isAll2All()){
            ff_a2a * a2a = (ff_a2a*) this->parentStructure;

            if (!processBB(a2a, in_C, out_C)){ // if the user has not wrapped the whole a2a, expan its sets

                for(ff_node* bb : a2a->getFirstSet())
                    processBB(bb, in_C, out_C);
                
                for(ff_node* bb : a2a->getSecondSet())
                    processBB(bb, in_C, out_C);
            }

        }
        
        // in/out nodes left to be added to the farm. The next lines does it
        for (const auto& pair : this->inout_){
            //std::cout << "Added INOUT node" << std::endl;
            this->add_workers({pair.second});
        }

        if (this->getNWorkers() == 0)
            return -1;

      
        // create receiver
        if (!isSource()){
            //std::cout << "Creating the receiver!" << std::endl;
            this->add_emitter(new ff_dreceiver(0 , this->endpoint, this->expectedInputConnections, buildRoutingTable(level1BB))); // set right parameters HERE!!
        }
        // create sender
        if (!isSink()){
            //std::cout << "Creating the sender!" << std::endl;
            this->add_collector(new ff_dsender(this->destinations), true);
        }
       

        //std::cout << "Built a farm of " << this->getNWorkers() << " workers!" << std::endl;
        // call the base class (ff_farm)'s prepare
        return 0;
    }

    ff_node* getWrapper(ff_node* n){
        return this->inout_[n];
    }

public:
    dGroup(ff_node* parent, std::string label): parentStructure(parent), endpoint(), destinations(), expectedInputConnections(0), in(this), out(this){
        dGroups::Instance()->addGroup(label, this);
    }

    int run(bool skip_init=false) override {return 0;}

    int run(ff_node* baseBB, bool skip_init=false) override {

        dGroups* groups_ = dGroups::Instance();
        groups_->parseConfig();
        
        buildFarm(reinterpret_cast<ff_pipeline*>(baseBB));

        return ff_farm::run(skip_init);
    }

    //int wait() {return ff_farm::wait();}


    void setEndpoint(const std::string address, const int port){
        this->endpoint.address = address;
        this->endpoint.port = port;
    }

    ff_endpoint getEndpoint(){return this->endpoint;}

    void setDestination(ff_endpoint e){ this->destinations.push_back(std::move(e));}

    void setExpectedInputConnections(int connections){this->expectedInputConnections = connections;}
    
    MySet<IN> in;
    MySet<OUT> out;
};


/**
 * If the user uses << operator -> serialization is used 
 * If the user uses <<= operator -> NO serialization is used
 **/

template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<(ff_node_t<Tin, Tout>* node){
    /*if (condizione){
        error("Errore!");
        throw 
    }*/
    
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else 
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, false, Tin, Tout>(node, 1, false, nullptr, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    } else
        this->group->in_.insert({node, {new WrapperIN<true, Tin, Tout>(node, false), true}});

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<(ff_minode_t<Tin, Tout>* node){
    /*if (condizione){
        error("Errore!");
        throw 
    }*/

    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){// the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, false, Tin, Tout>(node, 1, false, nullptr, reinterpret_cast<WrapperOUT<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getFirst())->getTransform())});
    } else
        this->group->in_.insert({node, {new WrapperIN<true, Tin, Tout>(node, false), true}});

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<(ff_monode_t<Tin, Tout>* node){
    
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){ 
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, false, Tin, Tout>(node, 1, false, nullptr, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    } else {
        ff_comb* combine = new ff_comb(new WrapperIN<true, Tin>(new ForwarderNode, true), node, true, false);
        this->group->in_.insert({node, {combine, true}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<(ff_node_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, false, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer())});
    } else
        this->group->out_.insert({node, {new WrapperOUT<true, Tin, Tout>(node, false), true}});

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<(ff_minode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, false, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer())});
    } else {
        ff_comb* combine = new ff_comb(node, new WrapperOUT<true, Tout>(new ForwarderNode, true), false, true);
        this->group->out_.insert({node, {combine, true}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<(ff_monode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<true, true, Tin, Tout>(node, 1, false)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, false, reinterpret_cast<WrapperIN<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getLast())->getFinalizer())});
    } else
        this->group->out_.insert({node, {new WrapperOUT<true, Tin, Tout>(node, false), true}});

    return *this;
}


template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<=(ff_node_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, nullptr, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    } else
        this->group->in_.insert({node, {new WrapperIN<false, Tin, Tout>(node, true), false}});

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<IN>& MySet<IN>::operator<<=(std::pair<ff_node_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    } else
        this->group->in_.insert({nodeFun.first, {new WrapperIN<false, Tin, Tout>(nodeFun.first, true, nodeFun.second), false}});

    return *this;
}


template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<=(ff_minode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, nullptr, reinterpret_cast<WrapperOUT<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getFirst())->getTransform())});
    
    } else
        this->group->in_.insert({node, {new WrapperIN<false, Tin, Tout>(node, true), false}});

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<IN>& MySet<IN>::operator<<=(std::pair<ff_minode_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(nodeFun.first, 1,  true, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second, reinterpret_cast<WrapperOUT<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getFirst())->getTransform())});
    
    } else
        this->group->in_.insert({nodeFun.first, {new WrapperIN<false, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second), false}});

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<IN>& MySet<IN>::operator<<=(ff_monode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(node);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, nullptr, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    
    } else {
        ff_comb* combine = new ff_comb(new WrapperIN<false, Tin, Tout>(new ForwarderNode, true), node, true);
        this->group->in_.insert({node, {combine, false}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<IN>& MySet<IN>::operator<<=(std::pair<ff_monode_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->out_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its output
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tout, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, true, Tin, Tout>(nodeFun.first, 1,  true, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second, ((WrapperOUT<false, Tin, Tout>*)handle.mapped().first)->getTransform())});
    
    } else {
        ff_comb* combine = new ff_comb(new WrapperIN<false, Tin, Tout>(new ForwarderNode, true, nodeFun.second), nodeFun.first, true);
        this->group->in_.insert({nodeFun.first, {combine, false}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<=(ff_node_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer())});
    } else
        this->group->out_.insert({node, {new WrapperOUT<false, Tin, Tout>(node, true), false}});

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<OUT>& MySet<OUT>::operator<<=(std::pair<ff_node_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(nodeFun.first, 1, true, nullptr, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer(), nodeFun.second)});
    } else
        this->group->out_.insert({nodeFun.first, {new WrapperOUT<false, Tin, Tout>(nodeFun.first, true, nodeFun.second), false}});

    return *this;
}



template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<=(ff_minode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer())});
    } else {
        ff_comb* combine = new ff_comb(node, new WrapperOUT<false, Tout>(new ForwarderNode, true), false, true);
        this->group->out_.insert({node, {combine, false}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<OUT>& MySet<OUT>::operator<<=(std::pair<ff_minode_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(nodeFun.first, 1, true, nullptr, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, ((WrapperIN<false, Tin, Tout>*)handle.mapped().first)->getFinalizer(), nodeFun.second)});
    } else {
        ff_comb* combine = new ff_comb(nodeFun.first, new WrapperOUT<false, Tout>(new ForwarderNode, true, nodeFun.second), false, true);
        this->group->out_.insert({nodeFun.first, {combine, false}});
    }

    return *this;
}

template<>
template<typename Tin, typename Tout>
MySet<OUT>& MySet<OUT>::operator<<=(ff_monode_t<Tin, Tout>* node){
    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(node);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({node, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(node, 1, true)});
            }
        } else
            this->group->inout_.insert({node, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(node, 1, true, reinterpret_cast<WrapperIN<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getLast())->getFinalizer())});
    } else
        this->group->out_.insert({node, {new WrapperOUT<false, Tin, Tout>(node, true), false}});

    return *this;
}

template<>
template<typename Tin, typename Tout, typename Function>
MySet<OUT>& MySet<OUT>::operator<<=(std::pair<ff_monode_t<Tin, Tout>*, Function> nodeFun){
    if (check_inout(nodeFun.first)) return *this; // the node is already processed in input and output, just skip it!

    auto handle = this->group->in_.extract(nodeFun.first);
    if (!handle.empty()){ // the node is edge also in its input
        if (handle.mapped().second){
            if constexpr (cereal::traits::is_output_serializable<Tin, cereal::BinaryOutputArchive>::value){
                this->group->inout_.insert({nodeFun.first, (ff_node*) new WrapperINOUT<true, false, Tin, Tout>(nodeFun.first, 1, true, nullptr, nodeFun.second)});
            }
        } else
            this->group->inout_.insert({nodeFun.first, (ff_node*)new WrapperINOUT<false, false, Tin, Tout>(nodeFun.first, 1, true, reinterpret_cast<WrapperIN<false, Tin, Tout>*>(reinterpret_cast<ff_comb*>(handle.mapped().first)->getLast())->getFinalizer(), nodeFun.second)});
    } else
        this->group->out_.insert({nodeFun.first, {new WrapperOUT<false, Tin, Tout>(nodeFun.first, 1, true, nodeFun.second), false}});

    return *this;
}

template<IOTypes T>
bool MySet<T>::check_inout(ff_node* node){
        return this->group->inout_.find(node) != this->group->inout_.end();
    }


void dGroups::parseConfig(){
        if (this->configFilePath.empty()) throw FF_Exception("Config file not defined!");

        std::ifstream is(this->configFilePath);

        if (!is) throw FF_Exception("Unable to open configuration file for the program!");

        std::vector<G> parsedGroups;

        try {
            cereal::JSONInputArchive ari(is);
            ari(cereal::make_nvp("groups", parsedGroups));
        } catch (const cereal::Exception& e){
            std::cerr << "Error parsing the JSON config file. Check syntax and structure of  the file and retry!" << std::endl;
            exit(EXIT_FAILURE);
        }

        for(G& g : parsedGroups)
            if (groups.find(g.name) != groups.end())
                reinterpret_cast<dGroup*>(groups[g.name])->setEndpoint(g.address, g.port);
            else {
                std::cout << "Cannot find group: " << g.name << std::endl;
                throw FF_Exception("A specified group in the configuration file has not been implemented! :(");
            }

        for(G& g : parsedGroups){
            dGroup* groupRef = reinterpret_cast<dGroup*>(groups[g.name]);
            for(std::string& conn : g.Oconn)
                if (groups.find(conn) != groups.end())
                    groupRef->setDestination(reinterpret_cast<dGroup*>(groups[conn])->getEndpoint());
                else throw FF_Exception("A specified destination has a wrong name! :(");
            
            groupRef->setExpectedInputConnections(expectedInputConnections(g.name, parsedGroups)); 
        }

    }

}

// redefinition of createGroup methods for ff_a2a and ff_pipeline
ff::dGroup& ff_a2a::createGroup(std::string name){
    dGroup * g = new dGroup(this, std::move(name));
    return *g;
}

ff::dGroup& ff_pipeline::createGroup(std::string name){
    dGroup * g = new dGroup(this, std::move(name));
    return *g;
}

// utility functions useful for creating suitable pairs to be used
// when defining custom serialization/deserialization functions
template<typename Tin, typename Tout, typename Function>
static inline std::pair<ff_node_t<Tin, Tout>*, Function> packup(ff_node_t<Tin, Tout>* node, Function f){
    return std::make_pair(node, f);
}
template<typename Tin, typename Tout, typename Function>
static inline std::pair<ff_minode_t<Tin, Tout>*, Function> packup(ff_minode_t<Tin, Tout>* node, Function f){
    return std::make_pair(node, f);
}
template<typename Tin, typename Tout, typename Function>
static inline std::pair<ff_monode_t<Tin, Tout>*, Function> packup(ff_monode_t<Tin, Tout>* node, Function f){
    return std::make_pair(node, f);
}

#endif
