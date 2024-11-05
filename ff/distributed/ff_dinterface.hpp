#ifndef FF_DINTERFACE_H
#define FF_DINTERFACE_H

#include <ff/ff.hpp>
#include <map>


#ifdef DFF_ENABLED
#include <ff/distributed/ff_dgroups.hpp>
#endif


namespace ff {

struct GroupInterface {
    std::string name;
    ff_node* base;
    GroupInterface(std::string name, ff_node* base = nullptr) : name(name), base(base){
#ifdef DFF_ENABLED
        if (!base) return;
        auto& annotated = dGroups::Instance()->annotated;
        if (annotated.find(base) == annotated.end())
            annotated[base] = name;
#endif
    }

    GroupInterface& operator<<(ff_node* node){
#ifdef DFF_ENABLED
        auto& annotated = dGroups::Instance()->annotated;
        
        if (base){
            // erase from the annotated the whole base building block
            annotated.erase(base);

            // check node figlio di base se base è diverso da nullptr
            //if (!getBB(base, node)){
            //    std::cerr << "Error in group " << name << ": a node outside the group base buildingblock was added. Please use static createGroup functions instead. Aborting.\n"; abort();
            //}
        }

        auto handler = annotated.find(node);
        if (handler == annotated.end())
            annotated[node] = name;
        else if (handler->second != name){
            std::cerr << "A node has been annotated in group " << name << " and in group " << handler->second << "! Aborting\n";
            abort();
        }
#endif
        return *this;
    }
	
    GroupInterface& operator<<(ff_node& node){
		return *this << &node;
	}   
};


GroupInterface ff_node::createGroup(const std::string& name){
#ifdef DFF_ENABLED
    dGroups::Instance()->annotateGroup(name, this);
#endif
    return GroupInterface(name, this);
}

GroupInterface createGroup(const std::string& name){
#ifdef DFF_ENABLED
    dGroups::Instance()->annotateGroup(name, nullptr);
#endif
    return GroupInterface(name);
}

} // namespace

#endif
