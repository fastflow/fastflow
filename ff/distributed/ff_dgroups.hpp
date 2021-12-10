#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <string>
#include <map>
#include <vector>
#include <sstream>

#include <getopt.h>

#include <ff/node.hpp>
#include <ff/utils.hpp>

#include <ff/distributed/ff_dprinter.hpp>

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#ifdef DFF_MPI
#include <mpi.h>
#endif

namespace ff {

enum Proto {TCP , MPI};

class dGroups {
public:

    static dGroups* Instance(){
        if (i == nullptr)
            i = new dGroups();
        return i;
    }

    Proto usedProtocol;

    void parseConfig(std::string);

    void consolidateGroups();

    void addGroup(std::string label, ff_node* g){ groups.insert(make_pair(label, g));}

    int size(){ return groups.size();}

    void setRunningGroup(std::string g){this->runningGroup = g;}
    
    void setRunningGroupByRank(int rank){
        this->runningGroup = parsedGroups[rank].name;
    }

	  const std::string& getRunningGroup() const { return runningGroup; }

    void forceProtocol(Proto p){this->usedProtocol = p;}

    bool isBuildByMyBuildingBlock(const std::string gName){ return groups[gName] == groups[runningGroup];}
	
    int run_and_wait_end(ff_node* parent){

        // perfrom connection between groups
        this->consolidateGroups();

        if (groups.find(runningGroup) == groups.end()){
            ff::error("The group specified is not found nor implemented!\n");
            return -1;
        }

        ff_node* runningGroup = this->groups[this->runningGroup];
        
        if (runningGroup->run(parent) < 0) return -1;
        if (runningGroup->wait() < 0) return -1;

      #ifdef DFF_MPI
        if (usedProtocol == Proto::MPI)
          if (MPI_Finalize() != MPI_SUCCESS) abort();
      #endif 
        
        return 0;
    }
protected:
    dGroups() : groups(), runningGroup() {
        // costruttore
    }

private:
    inline static dGroups* i = nullptr;
    std::map<std::string, ff_node*> groups;
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

    std::vector<G> parsedGroups;

    static inline std::vector<std::string> split (const std::string &s, char delim) {
        std::vector<std::string> result;
        std::stringstream ss (s);
        std::string item;

        while (getline (ss, item, delim))
            result.push_back (item);

        return result;
    }


    int expectedInputConnections(std::string groupName){
        int result = 0;
        for (const G& g : parsedGroups)
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
    
    dGroups::Instance()->parseConfig(configFile);

    if (!groupName.empty())
      dGroups::Instance()->forceProtocol(Proto::TCP);
  #ifdef DFF_MPI
    else
        dGroups::Instance()->forceProtocol(Proto::MPI);
  #endif


    if (dGroups::Instance()->usedProtocol == Proto::TCP){
       if (groupName.empty()){
        ff::error("Group not passed as argument!\nUse option --DFF_GName=\"group-name\"\n");
        return -1;
      } 
      dGroups::Instance()->setRunningGroup(groupName); 
    }

  #ifdef DFF_MPI
    if (dGroups::Instance()->usedProtocol == Proto::MPI){
      //MPI_Init(&argc, &argv);
      int provided;
      
      if (MPI_Init_thread(&argc, &argv, MPI_THREAD_MULTIPLE, &provided) != MPI_SUCCESS)
        return -1;
      
      
      // no thread support 
      if (provided < MPI_THREAD_MULTIPLE){
          error("No thread support by MPI\n");
          return -1;
      }

      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      dGroups::Instance()->setRunningGroupByRank(myrank);

      std::cout << "Running group: " << dGroups::Instance()->getRunningGroup() << " on rank: " <<  myrank << "\n";
    }

  #endif  

    // set the name for the printer
    ff::cout.setPrefix(dGroups::Instance()->getRunningGroup());

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
