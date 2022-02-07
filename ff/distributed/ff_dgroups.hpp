#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <algorithm>

#include <getopt.h>

#include <ff/ff.hpp>

#include <ff/distributed/ff_dprinter.hpp>
#include <ff/distributed/ff_dutils.hpp>
#include <ff/distributed/ff_dintermediate.hpp>

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
    friend struct GroupInterface;
    static dGroups* Instance(){
      static dGroups dg;
      return &dg;
    }
    
	/*~dGroups() {
		for (auto g : this->groups)
            if (g.second)
				delete g.second;
		groups.clear();
	}*/
	
    Proto usedProtocol;

    void parseConfig(std::string configFile){
      std::ifstream is(configFile);

        if (!is) throw FF_Exception("Unable to open configuration file for the program!");

        try {
            cereal::JSONInputArchive ari(is);
            ari(cereal::make_nvp("groups", this->parsedGroups));
            
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

    void annotateGroup(std::string name, ff_node* parentBB){
      if (annotatedGroups.contains(name)){
        std::cerr << "Group " << name << " created twice. Error!\n"; abort();
      }
      annotatedGroups[name].parentBB = parentBB;
    }

    int size(){ return annotatedGroups.size();}

    void setRunningGroup(std::string g){this->runningGroup = g;}
    
    void setRunningGroupByRank(int rank){
        this->runningGroup = parsedGroups[rank].name;
    }

	  const std::string& getRunningGroup() const { return runningGroup; }

    void forceProtocol(Proto p){this->usedProtocol = p;}

    bool isBuildByMyBuildingBlock(const std::string gName);
	
    int run_and_wait_end(ff_pipeline* parent){
        if (annotatedGroups.find(runningGroup) == annotatedGroups.end()){
            ff::error("The group specified is not found nor implemented!\n");
            return -1;
        }


      // qui dovrei creare la rappresentazione intermedia di tutti
      this->prepareIR(parent);

      // buildare il farm dalla rappresentazione intermedia del gruppo che devo rannare

      // rannere il farm come sotto!

        //ff_node* runningGroup = this->groups[this->runningGroup];
        
        //if (runningGroup->run(parent) < 0) return -1;
        //if (runningGroup->wait() < 0) return -1;

      #ifdef DFF_MPI
        if (usedProtocol == Proto::MPI)
          if (MPI_Finalize() != MPI_SUCCESS) abort();
      #endif 
        
        return 0;
    }
protected:
    dGroups() : runningGroup() {
        // costruttore
    }
    std::map<ff_node*, std::string> annotated;
private:
    inline static dGroups* i = nullptr;
    std::map<std::string, ff_IR> annotatedGroups;
    
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


    void prepareIR(ff_pipeline* parentPipe){
      ff::ff_IR& runningGroup_IR = annotatedGroups[this->runningGroup];
      ff_node* previousStage = getPreviousStage(parentPipe, runningGroup_IR.parentBB);
      ff_node* nextStage = getNextStage(parentPipe, runningGroup_IR.parentBB);
      // TODO: check coverage all 1st level

      for(size_t i = 0; i < parsedGroups.size(); i++){
        const G & g = parsedGroups[i];

        // throw an error if a group in the configuration has not been annotated in the current program
        if (!annotatedGroups.contains(g.name)) throw FF_Exception("present in the configuration file has not been implemented! :(");
        auto endpoint = this->usedProtocol == Proto::TCP ? ff_endpoint(g.address, g.port) : ff_endpoint(i);
        endpoint.groupName = g.name;

        // annotate the listen endpoint for the specified group
        annotatedGroups[g.name].listenEndpoint = endpoint;
      }
  

      /*auto g = std::find_if(parsedGroups.begin(), parsedGroups.end(), [this](const G& g){return g.name == getRunningGroup();});
      for(std::string& conn : g->Oconn)
          // build the list of outgoing connections for each group
          if (annotatedGroups.find(conn) != annotatedGroups.end())
              runningGroup_IR.destinationEndpoints.push_back(annotatedGroups[conn].listenEndpoint);
          else throw FF_Exception("A specified destination has a wrong name! :(");
      */
      
      // compute how many expected EOS the future running must wait before exiting
      /*size_t expectedEOS = 0;
      for (const G& g : parsedGroups)
        if (g.name != this->runningGroup)
          for (const std::string& conn : g.Oconn)
            if (conn == this->runningGroup) expectedEOS++;*/

      // build the map parentBB -> <groups names>
      std::map<ff_node*, std::set<std::string>> parentBB2GroupsName;
      for(auto& p: annotatedGroups) parentBB2GroupsName[p.second.parentBB].insert(p.first);

      // iterate over the 1st level building blocks and the list of create groups 
      for(const auto& pair : parentBB2GroupsName){

        //just build the current previous the current and the next stage of the parentbuilding block i'm going to exeecute  //// TODO: reprhase this comment!
        if (!(pair.first == previousStage || pair.first == runningGroup_IR.parentBB || pair.first == nextStage))
          continue;

        bool isSrc = isSource(pair.first, parentPipe);
        bool isSnk = isSink(pair.first, parentPipe);
        // check if from the under analysis 1st level building block it has been created just one group
        if (pair.second.size() == 1){
          // if the unique group is not the one i'm going to run just skip this 1st level building block
          if ((isSrc || pair.first->isDeserializable()) && (isSnk || pair.first->isSerializable())){
            auto& ir_ = annotatedGroups[*pair.second.begin()];
            ir_.insertInList(std::make_pair(pair.first, SetEnum::L), true);
            ir_.isSink = isSnk; ir_.isSource = isSrc;
          }
          continue; // skip anyway
        }
        
        // if i'm here it means that from this 1st level building block, multiple groups have been created! (An Error or an A2A or a Farm BB)
        std::set<std::pair<ff_node*, SetEnum>> children = getChildBB(pair.first);
        
        std::erase_if(children, [&](auto& p){
            if (!annotated.contains(p.first)) return false;
            std::string& groupName = annotated[p.first]; 
            if (((isSrc && p.second == SetEnum::L) || p.first->isDeserializable()) && (isSnk || p.first->isSerializable())){
              annotatedGroups[groupName].insertInList(std::make_pair(p.first, p.second)); return true;
            }
            return false;
        });

        if (!children.empty()){
          // seconda passata per verificare se ce da qualche parte c'è copertura completa altrimenti errore
          for(const std::string& gName : pair.second){
            ff_IR& ir = annotatedGroups[gName];
            ir.computeCoverage();
            if (ir.coverageL)
              std::erase_if(children, [&](auto& p){
                if (p.second == SetEnum::R && (isSnk || p.first->isSerializable()) && p.first->isDeserializable()){
                  ir.insertInList(std::make_pair(p.first, SetEnum::R)); return true;
                }
                return false;
              });

            if (ir.coverageR)
              std::erase_if(children, [&](auto& p){
                if (p.second == SetEnum::L && (isSrc || p.first->isDeserializable()) && p.first->isSerializable()){
                  ir.insertInList(std::make_pair(p.first, SetEnum::L)); return true;
                }
                return false;
              });
          }

          // ancora dei building block figli non aggiunti a nessun gruppo, lancio errore e abortisco l'esecuzione
          if (!children.empty()){
            std::cerr << "Some building block has not been annotated and no coverage found! You missed something. Aborting now" << std::endl;
            abort();
          }
        } else // compute the coverage anyway 
          for(const std::string& gName : pair.second) annotatedGroups[gName].computeCoverage();

      }

      //############# compute the number of excpected input connections
      if (runningGroup_IR.hasRightChildren()){
        auto& currentGroups = parentBB2GroupsName[runningGroup_IR.parentBB];
        runningGroup_IR.expectedEOS = std::count_if(currentGroups.cbegin(), currentGroups.cend(), [&](auto& gName){return (!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasLeftChildren());});
        // if the current group is horizontal count out itsleft from the all horizontals
        if (!runningGroup_IR.isVertical()) runningGroup_IR.expectedEOS -= 1;
      }
      // if the previousStage exists, count all the ouput groups pointing to the one i'm going to run
      if (previousStage && runningGroup_IR.hasLeftChildren())
        runningGroup_IR.expectedEOS += outputGroups(parentBB2GroupsName[previousStage]).size();

      if (runningGroup_IR.expectedEOS > 0) runningGroup_IR.hasReceiver = true;

      //############ compute the name of the outgoing connection groups
      if (runningGroup_IR.parentBB->isAll2All() && runningGroup_IR.isVertical() && runningGroup_IR.hasLeftChildren()){
          // inserisci tutte i gruppi di questo bb a destra
          for(const auto& gName: parentBB2GroupsName[runningGroup_IR.parentBB])
            if (!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasRightChildren())
              runningGroup_IR.destinationEndpoints.push_back(annotatedGroups[gName].listenEndpoint);
      } else {
        if (!runningGroup_IR.isVertical()){
          // inserisci tutti i gruppi come sopra
          for(const auto& gName: parentBB2GroupsName[runningGroup_IR.parentBB])
            if ((!annotatedGroups[gName].isVertical() || annotatedGroups[gName].hasRightChildren()) && gName != runningGroup)
              runningGroup_IR.destinationEndpoints.push_back(annotatedGroups[gName].listenEndpoint);
        }

        if (nextStage)
          for(const auto& gName : inputGroups(parentBB2GroupsName[nextStage]))
            runningGroup_IR.destinationEndpoints.push_back(annotatedGroups[gName].listenEndpoint);
      }

      if (!runningGroup_IR.destinationEndpoints.empty()) runningGroup_IR.hasSender = true;
      

      runningGroup_IR.buildIndexes();

      runningGroup_IR.print();
    }

  std::set<std::string> outputGroups(std::set<std::string> groupNames){
    if (groupNames.size() > 1)
      std::erase_if(groupNames, [this](const auto& gName){return (annotatedGroups[gName].isVertical() && annotatedGroups[gName].hasLeftChildren());});
    return groupNames;
  }

  std::set<std::string> inputGroups(std::set<std::string> groupNames){
    if (groupNames.size() > 1) 
      std::erase_if(groupNames,[this](const auto& gName){return (annotatedGroups[gName].isVertical() && annotatedGroups[gName].hasRightChildren());});
    return groupNames;
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

int ff_pipeline::run_and_wait_end() {
    dGroups::Instance()->run_and_wait_end(this);
    return 0;
}

int ff_a2a::run_and_wait_end(){
  ff_pipeline p;
  p.add_stage(this);
  dGroups::Instance()->run_and_wait_end(&p);
  return 0;
}



	
}
#endif
