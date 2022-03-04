#include <ff/ff.hpp>
#include <ff/distributed/ff_dgroups.hpp>
#include <map>

namespace ff {

struct GroupInterface {
    std::string name;
    GroupInterface(std::string name) : name(name){}

    GroupInterface& operator<<(ff_node* node){
        auto& annotated = dGroups::Instance()->annotated;
        auto handler = annotated.find(node);
        if (handler == annotated.end())
            annotated[node] = name;
        else if (handler->second != name){
            std::cerr << "Node has been annotated in group " << name << " and in group " << handler->second << "! Aborting\n";
            abort();
        }
        return *this;
    }
};


GroupInterface ff_node::createGroup(std::string name){
    dGroups::Instance()->annotateGroup(name, this);
    return GroupInterface(name);
}

}