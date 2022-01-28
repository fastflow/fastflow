#ifndef FF_DGROUP_H
#define FF_DGROUP_H

#include <ff/ff.hpp>
#include <getopt.h>
#include <iostream>
#include <fstream>
#include <map>
#include <set>
#include <exception>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_wrappers.hpp>
#include <ff/distributed/ff_dreceiver.hpp>
#include <ff/distributed/ff_dsender.hpp>

#ifdef DFF_MPI
#include <ff/distributed/ff_dreceiverMPI.hpp>
#include <ff/distributed/ff_dsenderMPI.hpp>
#endif

#include <ff/distributed/ff_dgroups.hpp>

#include <cereal/details/traits.hpp> // used for operators constexpr evaulations



namespace ff{

class dGroup;

enum IOTypes { IN, OUT };

template <IOTypes>
class MySet {
private:
    dGroup* group;
public:
    MySet(dGroup* group): group(group){ }

    MySet& operator<<(ff_node*);

    bool check_inout(ff_node* node);
};

class dGroups;

class dGroup : public ff_farm {

    friend class MySet<IN>;
    friend class MySet<OUT>;

    struct ForwarderNode : ff_node { 
        ForwarderNode(std::function<void(void*, dataBuffer&)> f){
            this->serializeF = f;
        }
        ForwarderNode(std::function<void*(dataBuffer&)> f){
            this->deserializeF = f;
        }
        void* svc(void* input){return input;}
    };
private:
    ff_node * parentStructure;
    ff_node * level1BB;

    ff_endpoint endpoint;
    std::vector<ff_endpoint> destinations;
    int expectedInputConnections;

    std::set<ff_node*> in_, out_, inout_;

    bool isSource(){return in_.empty() && inout_.empty();}
    bool isSink(){return out_.empty() && inout_.empty();}

    static bool isIncludedIn(const ff::svector<ff_node*>& firstSet, std::set<ff_node*>& secondSet){
        for (const ff_node* n : firstSet)
            if (std::find(secondSet.begin(), secondSet.end(), n) == secondSet.end())
                return false;
        return true;
    }

    ff_node* buildWrapper(ff_node* n, IOTypes t){
        if (t == IOTypes::IN){
            if (n->isMultiOutput())
                return  new ff_comb(new WrapperIN(new ForwarderNode(n->deserializeF), true), n, true, false);
            return new WrapperIN(n, 1, false);
        }

        if (t == IOTypes::OUT){
            int id = getIndexOfNode(level1BB, n, nullptr, IOTypes::OUT);
             if (n->isMultiInput())
                return new ff_comb(n, new WrapperOUT(new ForwarderNode(n->serializeF), id, 1, true), false, true);
            return new WrapperOUT(n, id, 1, false);
        }

        return nullptr;
    }

    bool replaceWrapper(const ff::svector<ff_node*>& list, IOTypes t){
        for (ff_node* node : list){
            ff_node* parentBB = getBB(this->parentStructure, node);
            if (parentBB != nullptr){
                
                ff_node* wrapped = buildWrapper(node, t);
                
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

    static inline bool isSeq(ff_node* n){return (!n->isAll2All() && !n->isComp() && !n->isFarm() && !n->isOFarm() && !n->isPipe());}

    bool processBB(ff_node* bb, std::set<ff_node*> in_C, std::set<ff_node*> out_C){
        if (isSeq(bb)){
            ff::svector<ff_node*> svectBB(1); svectBB.push_back(bb);
            if (isSource() && this->out_.find(bb) != this->out_.end() /*&& replaceWrapper(svectBB, this->out_)*/){
                this->add_workers({buildWrapper(bb, IOTypes::OUT)});
                return true;
            }

            if (isSink() && this->in_.find(bb) != this->in_.end() /*&& replaceWrapper(svectBB, this->in_)*/){
                this->add_workers({buildWrapper(bb, IOTypes::IN)});
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
        
        if ((isSource() || replaceWrapper(in_nodes, IOTypes::IN)) && (isSink() || replaceWrapper(out_nodes, IOTypes::OUT))){
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

    static ff::svector<ff_node*> getIONodes(ff_node* n, IOTypes t){
        ff::svector<ff_node*> sv;
        switch (t) {
            case IOTypes::IN:  n->get_in_nodes(sv);  break;
            case IOTypes::OUT: n->get_out_nodes(sv); break;
        }
        return sv;
    }

    /**
     * Retrieve the index of the node wrapper (or the alternative) in the original shared memory graph. 
     * Depending on the type t (IN - OUT) we get the index either in input or in output.
    */
    static int getIndexOfNode(ff_node* bb, ff_node* wrapper, ff_node* alternative, IOTypes t){
        if (bb->isAll2All()){
            ff_a2a* a2a = (ff_a2a*) bb;
            int index = 0;
            for (ff_node* n : a2a->getFirstSet()){
                for (const ff_node* io : getIONodes(n, t)){
                    if (io == wrapper || io == alternative)
                        return index;
                    index++;
                }
            }

            index = 0;
            for (ff_node* n : a2a->getSecondSet()){
                for (ff_node* io : getIONodes(n, t))
                    if (io == wrapper || io == alternative) 
                        return index; 
                    else index++;
            }
        }

        int index = 0;
        for (ff_node* io : getIONodes(bb, t))
            if (io == wrapper || io == alternative) 
                return index; 
            else index++;

        return 0;
    }

    /**
     * Given a node pointer w and an hint of the type of Wrapper t, get the poiinter of the original wrapped node. 
     */
    static ff_node* getOriginal(ff_node* w, IOTypes t){
        // if the wrapper is actually a composition, return the correct address of the orginal node!
        if (w->isComp()){
            switch (t){
                case IOTypes::IN: return reinterpret_cast<ff_comb*>(w)->getRight();
                case IOTypes::OUT: return reinterpret_cast<ff_comb*>(w)->getLeft();
                default: return nullptr;
            }
        }
        switch (t){
            case IOTypes::IN: return reinterpret_cast<internal_mi_transformer*>(w)->n;
            case IOTypes::OUT: return reinterpret_cast<internal_mo_transformer*>(w)->n;
            default: return nullptr;
        }
    }

    std::map<int, int> buildRoutingTable(ff_node* level1BB){
        std::map<int, int> routingTable;
        int localIndex = 0;
        for (ff_node* inputBB : this->getWorkers()){
            ff::svector<ff_node*> inputs; inputBB->get_in_nodes(inputs);
            for (ff_node* input : inputs)
                routingTable[getIndexOfNode(level1BB, input, getOriginal(input, IOTypes::IN), IOTypes::IN)] = localIndex++;
        }
        return routingTable;
    }

    int buildFarm(ff_pipeline* basePipe = nullptr){

        // find the 1 level builiding block which containes the group (level 1 BB means a BB which is a stage in the main piepline)
        //ff_node* level1BB = this->parentStructure;
        while(!isStageOf(level1BB, basePipe)){
            level1BB = getBB(basePipe, level1BB);
            if (!level1BB || level1BB == basePipe) throw FF_Exception("A group were created from a builiding block not included in the Main Pipe! :(");
        }

        int onDemandQueueLength = 0; bool onDemandReceiver = false; bool onDemandSender = false;
        
        if (this->parentStructure->isPipe())
           processBB(this->parentStructure, in_, out_);

        if (this->parentStructure->isAll2All()){
            ff_a2a * a2a = (ff_a2a*) this->parentStructure;

            if (!processBB(a2a, in_, out_)){ // if the user has not wrapped the whole a2a, expand its sets
                bool first = false, second = false;
                
                for(ff_node* bb : a2a->getFirstSet())
                    if (processBB(bb, in_, out_))
                        first = true;
                
                for(ff_node* bb : a2a->getSecondSet())
                    if (processBB(bb, in_, out_))
                        second = true;
                
                // check on input/output nodes, used for ondemand stuff and for checking collision between nodes
                if (!first && !second){
                        for (const auto& n : this->inout_){
                            if (std::find(a2a->getFirstSet().begin(), a2a->getFirstSet().end(), n) != a2a->getFirstSet().end())
                                first = true;
                            else if (std::find(a2a->getSecondSet().begin(), a2a->getSecondSet().end(), n) != a2a->getSecondSet().end())
                                second = true;
                        }
                    }
                
                // if the ondemand scheduling is set in the a2a, i need to adjust the queues of this farm in order to implement the ondemand policy
                if (a2a->ondemand_buffer() > 0){

                    onDemandQueueLength = a2a->ondemand_buffer();
                    if (first) {this->setOutputQueueLength(1, true); onDemandSender = true;} // always set to 1 the length of the queue between worker and collector (SOURCE side)
                    if (second) {this->set_scheduling_ondemand(a2a->ondemand_buffer()); onDemandReceiver = true;} // set the right length of the queue between emitter and worker (SINK side)
                }

                if (first && second) throw FF_Exception("Nodes from first and second of an A2A cannot belong to the same group!!");
            }

        }
        
        // in/out nodes left to be added to the farm. The next lines does it
        for (ff_node* n : this->inout_){
            //std::cout << "Added INOUT node" << std::endl;
            this->add_workers({new WrapperINOUT(n, getIndexOfNode(level1BB, n, nullptr, IOTypes::OUT))});
        }

        if (this->getNWorkers() == 0)
            return -1;
      
        // create receiver
        Proto currentProto = dGroups::Instance()->usedProtocol;
        if (!isSource()){
            if (currentProto == Proto::TCP){
                if (onDemandReceiver)
                    this->add_emitter(new ff_dreceiverOD(this->endpoint, this->expectedInputConnections, buildRoutingTable(level1BB))); // set right parameters HERE!!
                else
                    this->add_emitter(new ff_dreceiver(this->endpoint, this->expectedInputConnections, buildRoutingTable(level1BB)));
            }

        #ifdef DFF_MPI
            if (currentProto == Proto::MPI){
                if (onDemandReceiver)
                    this->add_emitter(new ff_dreceiverMPIOD(this->expectedInputConnections, buildRoutingTable(level1BB))); // set right parameters HERE!!
                else
                    this->add_emitter(new ff_dreceiverMPI(this->expectedInputConnections, buildRoutingTable(level1BB)));
            }
        #endif

			this->cleanup_emitter();
        }
        // create sender
        if (!isSink()){
            
            if (currentProto == Proto::TCP){
                if (onDemandSender)
                    this->add_collector(new ff_dsenderOD(this->destinations, onDemandQueueLength), true);
                else
                    this->add_collector(new ff_dsender(this->destinations), true);
            }

        #ifdef DFF_MPI
            if (currentProto == Proto::MPI){
                if (onDemandSender)
                    this->add_collector(new ff_dsenderMPIOD(this->destinations, onDemandQueueLength), true);
                else
                    this->add_collector(new ff_dsenderMPI(this->destinations), true);
            }
        #endif

        }
       
        return 0;
    }

public:
    dGroup(ff_node* parent, std::string label): parentStructure(parent), level1BB(parent), endpoint(), destinations(), expectedInputConnections(0), in(this), out(this){
        dGroups::Instance()->addGroup(label, this);
    }

	~dGroup() {
        // TO DO: check the cleanup with the new way of generate wrapping!
		/*if (!prepared) {
			for(auto s: in_)
				delete s.second;
			for(auto s: out_)
				delete s.second;
			for(auto s: inout_)
				delete s.second;
		}
        */
	}
	
    int run(bool skip_init=false) override {return 0;}

    int run(ff_node* baseBB, bool skip_init=false) override {
        buildFarm(reinterpret_cast<ff_pipeline*>(baseBB));
		return ff_farm::run(skip_init);
    }


    void setEndpoint(ff_endpoint e){
        this->endpoint = e;
    }

    ff_endpoint getEndpoint(){return this->endpoint;}

	ff_node* getParent() const { return parentStructure; }
	
    void setDestination(ff_endpoint e){ this->destinations.push_back(std::move(e));}

    void setExpectedInputConnections(int connections){this->expectedInputConnections = connections;}
    
    MySet<IN> in;
    MySet<OUT> out;
};

template<>
MySet<IN>& MySet<IN>::operator<<(ff_node* node){
    if (!node->isDeserializable()){
        error("The annotated node is not able to deserialize the type!\n");
        abort();
    }

    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    if (!this->group->out_.extract(node).empty()) // if the node was also annotedted in output just create the WrapperINOUT
        this->group->inout_.insert(node);
    else
        this->group->in_.insert(node);

    return *this;
}

template<>
MySet<OUT>& MySet<OUT>::operator<<(ff_node* node){
    if (!node->isSerializable()){
        error("The annotated node is not able to serialize the type!\n");
        abort();
    }

    if (check_inout(node)) return *this; // the node is already processed in input and output, just skip it!

    if (!this->group->in_.extract(node).empty()) // if the node was also annotedted in output just create the WrapperINOUT
        this->group->inout_.insert(node);
    else
        this->group->out_.insert(node);

    return *this;
}

template<IOTypes T>
bool MySet<T>::check_inout(ff_node* node){
        return this->group->inout_.find(node) != this->group->inout_.end();
    }

void dGroups::consolidateGroups(){
    
}

bool dGroups::isBuildByMyBuildingBlock(const std::string gName) {
	auto g1 = reinterpret_cast<dGroup*>(groups[gName]);
	auto g2 = reinterpret_cast<dGroup*>(groups[runningGroup]);
	
	return g1->getParent() == g2->getParent();
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


#endif
