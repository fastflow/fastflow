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

    struct ForwarderNode : ff_node{ 
        ForwarderNode(std::function<void(void*, dataBuffer&)> f){
            this->serializeF = f;
        }
        ForwarderNode(std::function<void*(dataBuffer&)> f){
            this->deserializeF = f;
        }
        void* svc(void* input){return input;}
    };
public:
    MySet(dGroup* group): group(group){ }

    MySet& operator<<(ff_node*);

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
	bool executed = false;
    /**
     * Key: reference to original node
     * Value: pair of [reference to wrapper, serialization_required] 
     **/
    std::map<ff_node*, ff_node*> in_, out_;
    std::map<ff_node*, ff_node*> inout_;

    bool isSource(){return in_.empty() && inout_.empty();}
    bool isSink(){return out_.empty() && inout_.empty();}

    static bool isIncludedIn(const ff::svector<ff_node*>& firstSet, std::vector<ff_node*>& secondSet){
        for (const ff_node* n : firstSet)
            if (std::find(secondSet.begin(), secondSet.end(), n) == secondSet.end())
                return false;
        return true;
    }

    bool replaceWrapper(const ff::svector<ff_node*>& list, std::map<ff_node*, ff_node*>& wrappers_){
        for (ff_node* node : list){
            ff_node* parentBB = getBB(this->parentStructure, node);
            if (parentBB != nullptr){
                
                ff_node* wrapped = wrappers_[node];
                
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
        auto resultI = std::find_if(this->in_.begin(), this->in_.end(), [&](const std::pair<ff_node*,ff_node*> &pair){return pair.second == wrapper;});
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
                this->add_workers({this->out_[bb]});
                return true;
            }

            if (isSink() && this->in_.find(bb) != this->in_.end() && replaceWrapper(svectBB, this->in_)){
                this->add_workers({this->in_[bb]});
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

        int onDemandQueueLength = 0; bool onDemandReceiver = false; bool onDemandSender = false;

        if (this->parentStructure->isPipe())
           processBB(this->parentStructure, in_C, out_C);

        if (this->parentStructure->isAll2All()){
            ff_a2a * a2a = (ff_a2a*) this->parentStructure;

            if (!processBB(a2a, in_C, out_C)){ // if the user has not wrapped the whole a2a, expand its sets
                bool first = false, second = false;
                
                for(ff_node* bb : a2a->getFirstSet())
                    if (processBB(bb, in_C, out_C))
                        first = true;
                
                for(ff_node* bb : a2a->getSecondSet())
                    if (processBB(bb, in_C, out_C))
                        second = true;
                
                // check on input/output nodes, used for ondemand stuff and for checking collision between nodes
                if (!first && !second){
                        for (const auto& pair : this->inout_){
                            if (std::find(a2a->getFirstSet().begin(), a2a->getFirstSet().end(), pair.first) != a2a->getFirstSet().end())
                                first = true;
                            else if (std::find(a2a->getSecondSet().begin(), a2a->getSecondSet().end(), pair.first) != a2a->getSecondSet().end())
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
        for (const auto& pair : this->inout_){
            //std::cout << "Added INOUT node" << std::endl;
            this->add_workers({pair.second});
        }

        if (this->getNWorkers() == 0)
            return -1;
      
        // create receiver
        Proto currentProto = dGroups::Instance()->usedProtocol;
        if (!isSource()){
            if (currentProto == Proto::TCP){
                if (onDemandReceiver)
                    this->add_emitter(new ff_dreceiverOD(this->endpoint, this->expectedInputConnections, 0, buildRoutingTable(level1BB))); // set right parameters HERE!!
                else
                    this->add_emitter(new ff_dreceiver(this->endpoint, this->expectedInputConnections, 0, buildRoutingTable(level1BB)));
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

    ff_node* getWrapper(ff_node* n){
        return this->inout_[n];
    }

public:
    dGroup(ff_node* parent, std::string label): parentStructure(parent), endpoint(), destinations(), expectedInputConnections(0), in(this), out(this){
        dGroups::Instance()->addGroup(label, this);
    }

	~dGroup() {
		if (!executed) {
			for(auto s: in_)
				delete s.second;
			for(auto s: out_)
				delete s.second;
			for(auto s: inout_)
				delete s.second;
		}
	}
	
    int run(bool skip_init=false) override {return 0;}

    int run(ff_node* baseBB, bool skip_init=false) override {
        buildFarm(reinterpret_cast<ff_pipeline*>(baseBB));
		auto r= ff_farm::run(skip_init);
		executed = true;
        return r;
    }


    void setEndpoint(ff_endpoint e){
        this->endpoint = e;
    }

    ff_endpoint getEndpoint(){return this->endpoint;}

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
        this->group->inout_.emplace(node, new WrapperINOUT(node, 1, false));
    else {
        if (node->isMultiOutput())
            this->group->in_.emplace(node,  new ff_comb(new WrapperIN(new ForwarderNode(node->deserializeF), true), node, true, false));
        else
            this->group->in_.emplace(node, new WrapperIN(node, 1, false));
    }

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
        this->group->inout_.emplace(node, new WrapperINOUT(node, 1, false));
    else {
        if (node->isMultiInput())
            this->group->out_.emplace(node, new ff_comb(node, new WrapperOUT(new ForwarderNode(node->serializeF), 1, true), false, true));
        else
            this->group->out_.emplace(node, new WrapperOUT(node, 1, false));
    }

    return *this;
}

template<IOTypes T>
bool MySet<T>::check_inout(ff_node* node){
        return this->group->inout_.find(node) != this->group->inout_.end();
    }

void dGroups::consolidateGroups(){
    for(size_t i = 0; i < parsedGroups.size(); i++){
        const G & g = parsedGroups[i];
        if (groups.find(g.name) != groups.end())
            switch (this->usedProtocol){
                case Proto::TCP : reinterpret_cast<dGroup*>(groups[g.name])->setEndpoint(ff_endpoint(g.address, g.port)); break;
                case Proto::MPI : reinterpret_cast<dGroup*>(groups[g.name])->setEndpoint(ff_endpoint(i)); break;
            }
        else {
            std::cout << "Cannot find group: " << g.name << std::endl;
            throw FF_Exception("A specified group in the configuration file has not been implemented! :(");
        }
    }


    for(G& g : parsedGroups){
            dGroup* groupRef = reinterpret_cast<dGroup*>(groups[g.name]);
            for(std::string& conn : g.Oconn)
                if (groups.find(conn) != groups.end())
                    groupRef->setDestination(reinterpret_cast<dGroup*>(groups[conn])->getEndpoint());
                else throw FF_Exception("A specified destination has a wrong name! :(");
            
            groupRef->setExpectedInputConnections(this->expectedInputConnections(g.name)); 
        }

}

void dGroups::parseConfig(std::string configFile){

        std::ifstream is(configFile);

        if (!is) throw FF_Exception("Unable to open configuration file for the program!");

        try {
            cereal::JSONInputArchive ari(is);
            ari(cereal::make_nvp("groups", parsedGroups));
            
            // get the protocol to be used from the configuration file
            try {
                std::string tmpProtocol;
                ari(cereal::make_nvp("protocol", tmpProtocol));
                if (tmpProtocol == "MPI"){
                    #ifdef DFF_MPI
                        this->usedProtocol = Proto::MPI;
                    #else
                        std::cout << "NO MPI support! Falling back to TCP\n";
                        this->usedProtocol = Proto::TCP;
                    #endif 

                } else this->usedProtocol = Proto::TCP;
            } catch (cereal::Exception&) {
                ari.setNextName(nullptr);
                this->usedProtocol = Proto::TCP;
            }

        } catch (const cereal::Exception& e){
            std::cerr << "Error parsing the JSON config file. Check syntax and structure of the file and retry!" << std::endl;
            exit(EXIT_FAILURE);
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
#endif
