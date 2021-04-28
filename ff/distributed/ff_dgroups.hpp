#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <string>
#include <map>
#include <vector>
#include <sstream>

#include <getopt.h>

#include <ff/node.hpp>
#include <ff/utils.hpp>

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

	const std::string& getRunningGroup() const { return runningGroup; }
	
    int run_and_wait_end(ff_node* parent){
        if (groups.find(runningGroup) == groups.end()){
            ff::error("The group specified is not found nor implemented!\n");
            return -1;
        }

        ff_node* runningGroup = this->groups[this->runningGroup];
        
        if (runningGroup->run(parent) < 0) return -1;
        if (runningGroup->wait() < 0) return -1;
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

static inline int DFF_Init(int& argc, char**& argv){
    
    std::string configFile, groupName;

    for(int i = 0; i < argc; i++){
      if (strstr(argv[i], "--DFF_Config") != NULL){
        char * equalPosition = strchr(argv[i], '=');
        if (equalPosition == NULL){
          // the option is in the next argument array position
          configFile = std::string(argv[i+1]);
          argv[i] = argv[i+1] = NULL;
          i++;
        } else {
          // the option is in the next position of this string
          configFile = std::string(++equalPosition);
          argv[i] = NULL;
        }
        continue;
      }

      if (strstr(argv[i], "--DFF_GName") != NULL){
        char * equalPosition = strchr(argv[i], '=');
        if (equalPosition == NULL){
          // the option is in the next argument array position
          groupName = std::string(argv[i+1]);
          argv[i] = argv[i+1] = NULL;
          i++;
        } else {
          // the option is in the next position of this string
          groupName = std::string(++equalPosition);
          argv[i] = NULL;
        }
        continue;
      } 
    }

    if (configFile.empty()){
      ff::error("Config file not passed as argument!\nUse option --DFF_Config=\"config-file-name\"\n");
      return -1;
    }

    if (groupName.empty()){
      ff::error("Group not passed as argument!\nUse option --DFF_GName=\"group-name\"\n");
      return -1;
    }

    dGroups::Instance()->setRunningGroup(groupName); 
    dGroups::Instance()->setConfigFile(configFile);


    // recompact the argv array
    int j = 0;
    for(int i = 0;  i < argc; i++)
      if (argv[i] != NULL)
        argv[j++] = argv[i];
    
    // update the argc value
    argc = j;

    return 0;
}

static inline const std::string DFF_getMyGroup() {
	return dGroups::Instance()->getRunningGroup();
}
	
}

#endif
