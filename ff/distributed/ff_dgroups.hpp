#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <string>
#include <map>
#include <vector>
#include <sstream>

#include <ff/node.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>
namespace ff {

class dGroups {
public:

    static dGroups* Instance(){
        if (i == nullptr)
            i = new dGroups();
        return i;
    }

    void parseConfig();

    void addGroup(std::string label, ff_node* g){ groups.insert(make_pair(label, g));}

    int size(){ return groups.size();}

    void setConfigFile(std::string f){this->configFilePath = f;}

    void setRunningGroup(std::string g){this->runningGroup = g;}

    int run_and_wait_end(ff_node* parent){
        ff_node* runningGroup = this->groups[this->runningGroup];
        runningGroup->run(parent);
        runningGroup->wait();
        return 0;
    }
protected:
    dGroups() : groups(), configFilePath(), runningGroup() {
        // costruttore
    }

private:
    inline static dGroups* i = nullptr;
    std::map<std::string, ff_node*> groups;
    std::string configFilePath;
    std::string runningGroup;

    // helper class to parse config file Json
    struct G {
        std::string name;
        std::string address;
        int port;
        std::vector<std::string> Oconn;

        template <class Archive>
        void load( Archive & ar ){
            ar(cereal::make_nvp("name", name));
            
            try {
                std::string endpoint;
                ar(cereal::make_nvp("endpoint", endpoint)); std::vector endp(split(endpoint, ':'));
                address = endp[0]; port = std::stoi(endp[1]);
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

            try {
                ar(cereal::make_nvp("OConn", Oconn));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}
        }
    };

    static inline std::vector<std::string> split (const std::string &s, char delim) {
        std::vector<std::string> result;
        std::stringstream ss (s);
        std::string item;

        while (getline (ss, item, delim))
            result.push_back (item);

        return result;
    }


    static int expectedInputConnections(std::string groupName, std::vector<G>& groups){
        int result = 0;
        for (const G& g : groups)
            if (g.name != groupName)
                for (const std::string& conn : g.Oconn)
                    if (conn == groupName)  result++;
        return result;
    }
};

}

#endif