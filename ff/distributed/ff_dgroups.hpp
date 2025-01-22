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
 */
/* Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FF_DGROUPS_H
#define FF_DGROUPS_H

#include <signal.h>
#include <getopt.h>

#include <string>
#include <map>
#include <set>
#include <vector>
#include <sstream>
#include <algorithm>

#include <ff/ff.hpp>

#include <ff/distributed/ff_ddefines.hpp>
#include <ff/distributed/ff_dprinter.hpp>
#include <ff/distributed/ff_dutils.hpp>
#include <ff/distributed/ff_dintermediate.hpp>
#include <ff/distributed/ff_dgroup.hpp>

#include "MTCL/include/mtcl.hpp"

#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#ifdef DFF_MPI
#include <mpi.h>
#endif

namespace ff {

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
    std::string baseProtocol = "TCP";

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
                this->baseProtocol = tmpProtocol;
                /*if (tmpProtocol == "MPI"){
                    #ifdef DFF_MPI
                        this->usedProtocol = Proto::MPI;
                    #else
                        std::cout << "NO MPI support! Falling back to TCP\n";
                        this->usedProtocol = Proto::TCP;
                    #endif 

                } else this->usedProtocol = Proto::TCP;
                */
            } catch (cereal::Exception&) {
                ari.setNextName(nullptr);
                //this->usedProtocol = Proto::TCP;
                this->baseProtocol = "TCP";
            }

        } catch (const cereal::Exception& e){
            std::cerr << "Error parsing the JSON config file. Check syntax and structure of the file and retry!" << std::endl;
            exit(EXIT_FAILURE);
        }
    }

    void annotateGroup(const std::string& name, ff_node* parentBB){
      if (annotatedGroups.count(name)){
        std::cerr << "Group " << name << " created twice. Error!\n"; abort();
      }
      annotatedGroups.insert(name);
    }

    int size(){ return annotatedGroups.size();}

    void setRunningGroup(std::string g){this->runningGroup = g;}
    
    void setRunningGroupByRank(int rank){
        this->runningGroup = parsedGroups[rank].name;
    }

    /*
    * Set the thread mapping if specified in the configuration file. Otherwise use the default mapping specified in the legacy FastFlow config.hpp file.
    * In config file the mapping can be specified for each group through the key "threadMapping"
    */
    void setThreadMapping(){
      auto g = std::find_if(parsedGroups.begin(), parsedGroups.end(), [this](auto& g){return g.name == this->runningGroup;});
      if (g != parsedGroups.end() && !g->threadMapping.empty())
        threadMapper::instance()->setMappingList(g->threadMapping.c_str());
    }

	  const std::string& getRunningGroup() const { return runningGroup; }

    //void forceProtocol(Proto p){this->usedProtocol = p;}
	
    int run_and_wait_end(ff_pipeline* parent){
        if (annotatedGroups.find(runningGroup) == annotatedGroups.end()){
            ff::error("The group %s is not found nor implemented!\n", runningGroup.c_str());
            return -1;
        }

      this->prepareIR2(parent);

#ifdef DFF_PRINT_IR
      runningIR.print();
#endif
      MessageAllocator::init(MESSAGE_PREALLOCATE);

      // buildare il farm dalla rappresentazione intermedia del gruppo che devo rannare
      dGroup _grp(runningIR);
      if (_grp.run() < 0){
        std::cerr << "Error running the group!" << std::endl;
        return -1;
      }

      if (_grp.wait() < 0){
        std::cerr << "Error waiting the group!" << std::endl;
        return -1;
      }

      MessageAllocator::finalize();
      
      return 0;
    }

protected:
    dGroups() : runningGroup() {
        // costruttore
    }
    std::map<ff_node*, std::string> annotated;
private:
    std::set<std::string> annotatedGroups;
    
    std::string runningGroup;
    ff_IR_V2 runningIR;

    // helper class to parse config file Json
    struct G {
        inline static int rankCounter = 0;
    public:
        std::string name;
        std::string address;
        std::string threadMapping;
        int port;
        int batchSize          = DEFAULT_BATCH_SIZE;
        size_t batchByteSize   = DEFAULT_BATCH_BYTE_SIZE;
        int messageOTF         = DEFAULT_MESSAGE_OTF;
        int rank;

        template <class Archive>
        void load( Archive & ar ){
            rank = rankCounter++;
            ar(cereal::make_nvp("name", name));
            
            try {
                std::string endpoint;
                ar(cereal::make_nvp("endpoint", endpoint)); std::vector endp(split(endpoint, ':'));
                address = endp[0]; port = std::stoi(endp[1]);
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

            try {
                ar(cereal::make_nvp("batchSize", batchSize));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

             try {
              std::string batchByteSize_str;
                ar(cereal::make_nvp("batchByteSize", batchByteSize_str));
                batchByteSize = convertToBytes(batchByteSize_str);
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}
              catch (std::invalid_argument& e){std::cerr << "Error parsing batchByteSize option: " << e.what() << std::endl;}

             try {
                ar(cereal::make_nvp("messageOTF", messageOTF));
            } catch (cereal::Exception&) {ar.setNextName(nullptr);}

            try {
                ar(cereal::make_nvp("threadMapping", threadMapping));
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

    inline ChannelLocality getLocality(std::map<ff_node*, std::string*>& map, ff_node* n){
      return (*map[n] == this->runningGroup ? ChannelLocality::LOCAL : ChannelLocality::REMOTE);
    }

    inline std::string buildEndpointString(const G& g){
      if (this->baseProtocol == "MPI") 
          return "MPI:" + std::to_string(g.rank);
        else 
          return this->baseProtocol + ":" + g.address + ":" + std::to_string(g.port);
    }

    void prepareIR2(ff_node* root){

        auto parsedRunningGroup_it = std::find_if(parsedGroups.begin(), parsedGroups.end(), [this](G& g){return g.name == this->runningGroup;});
        if (parsedRunningGroup_it == parsedGroups.end()){
          std::cerr << "The running group <" << this->runningGroup << "> is not present in configuration file! Aborting\n"; abort();
        }

        // set the ids of the nodes for the whole original application
        setIDs(root, "");

        std::vector<ff_node*> nodesInGroup;
        // set used to compute the names of groups this group connect to (egress connections), and names of groups will connect to this group (ingress connections)
        std::map<std::string, std::vector<ff_node*>> ingressRemoteGroups, egressRemoteGroups;

        bool groupWrappedAround = false;

        // compute the expanded list of nodes in groups
        std::map<ff_node*, std::string*> expandedAnnotated;
        for(auto& [node, g] : this->annotated){
          ff::svector<ff_node*> tmp;
          node->get_in_nodes(tmp); node->get_out_nodes(tmp);
          for(ff_node* n : tmp)
            expandedAnnotated.emplace(n, &g);
        }

        for(auto& [node, g] : this->annotated)
          if (g == this->runningGroup){
            nodesInGroup.push_back(node);

            IngressEgressChannels_t channelsDescription;

            {
              ff::svector<ff_node*> inputs; node->get_out_nodes(inputs); // maybe we should add also get_in_nodes_feedback
              for(ff_node* node_ : inputs) {
                IngressEgressChannels_t& cd = runningIR.channelsDictionary[node_];
                // retrive all the nodes connected in output of the current node
                auto&& consumers = getConsumers(node_, root);


                // scan the feedback consumers channels
                for (ff_node* c : std::get<1>(consumers)) {
                  auto loc = getLocality(expandedAnnotated, c);
                  cd.second.emplace_back(c, ChannelType::FBK, loc, -1);
                  // if the destination is remote add the remote group name to the set of egressRemoteGroups
                  if (loc == ChannelLocality::REMOTE) 
                    egressRemoteGroups[*expandedAnnotated[c]].push_back(c);
                  else 
                    groupWrappedAround = true;
                }

                int queue_length = std::get<2>(consumers) > 0 ? std::get<2>(consumers) : -1;
                // scan the forward consumers channels
                for (ff_node* c : std::get<0>(consumers)) {
                  auto loc = getLocality(expandedAnnotated, c);
                  cd.second.emplace_back(c, ChannelType::FWD, loc, queue_length);

                  // if the destination is remote add the remote group name to the set of egressRemoteGroups
                  if (loc == ChannelLocality::REMOTE) 
                    egressRemoteGroups[*expandedAnnotated[c]].push_back(c);
                }
              }
            }

            {
              ff::svector<ff_node*> outputs; node->get_in_nodes(outputs); // maybe we should add also get_in_nodes_feedback
              for(ff_node* node_ : outputs) {
                IngressEgressChannels_t& cd = runningIR.channelsDictionary[node_];

                // retrive all the nodes connected in input to the current node
                auto&& feeders = getFeeders(node_, root);

                 // scan the feedback consumers channels
                for (ff_node* c : std::get<1>(feeders)) {
                  auto loc = getLocality(expandedAnnotated, c);
                  cd.first.emplace_back(c, ChannelType::FBK, loc, -1);
                  // if the destination is remote add the remote group name to the set of egressRemoteGroups
                  if (loc == ChannelLocality::REMOTE) 
                    ingressRemoteGroups[*expandedAnnotated[c]].push_back(c);
                }

                int queue_length = std::get<2>(feeders) > 0 ? std::get<2>(feeders) : -1;

                // scan the forward feeders channel
                for (ff_node* c : std::get<0>(feeders)) {
                  auto loc = getLocality(expandedAnnotated, c);
                  cd.first.emplace_back(c, ChannelType::FWD, loc, queue_length);

                  // if the destination is remote add the remote group name to the set of egressRemoteGroups
                  if (loc == ChannelLocality::REMOTE) 
                    ingressRemoteGroups[*expandedAnnotated[c]].push_back(c);
                } 
              }
            }
          }

        // sort the list of nodes in this group and build the representation of the A2A representing the group, and then save it in the IR
        sortBasedOnLabel(nodesInGroup);
        runningIR.bucketsDistribution = build_A2A_structure(nodesInGroup);

        // populate the list of egress endpoint and the name of input groups
        // egress connections
        for(auto& [n, dest_v] : egressRemoteGroups) {
          auto parsedGroup_it = std::find_if(parsedGroups.begin(), parsedGroups.end(), [&](G& e){return e.name == n;});
          if (parsedGroup_it == parsedGroups.end()){
            std::cerr << "The group <" << n << "> is not present in the configuration file! Aborting!";
          }
          runningIR.destinationEndpoints.emplace_back(n, buildEndpointString(*parsedGroup_it), dest_v);
        }

        // ingress connections
        for(auto& [n, _] : ingressRemoteGroups) runningIR.ingressRemoteConnectionsGroupsName.push_back(n);

        // set the listening endpoint for the running group 
        runningIR.listeningEndpoint = buildEndpointString(*parsedRunningGroup_it);

        runningIR.batchSize = parsedRunningGroup_it->batchSize;
        runningIR.batchByteSize = parsedRunningGroup_it->batchByteSize;

        runningIR.messageOTF = parsedRunningGroup_it->messageOTF;

        runningIR.groupWrappedAround = groupWrappedAround;

    }


};



static inline int DFF_Init(int& argc, char**& argv){
    struct sigaction s;
    memset(&s,0,sizeof(s));    
    s.sa_handler=SIG_IGN;
    if ( (sigaction(SIGPIPE,&s,NULL) ) == -1 ) {   
      perror("sigaction");
      return -1;
    } 


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

#if defined(DFF_MPI) || defined(ENABLE_MPI)
    if (groupName.empty()) {
      if (dGroups::Instance()->baseProtocol != "MPI")
        ff::error("Falling back to MPI since no group name passed as argument\n");
      dGroups::Instance()->baseProtocol = "MPI";
    }  
#else
    if (dGroups::Instance()->baseProtocol == "MPI") {
        ff::error("MPI support disabled during compilation! Recompile with MPI and retry\n");
        return -1;
    }
    if (groupName.empty()) {
      ff::error("Group not passed as argument!\nUse option --DFF_GName=\"group-name\"\n");
      return -1;
    }
#endif
    else {
      // set the running group
      dGroups::Instance()->setRunningGroup(groupName);
    }

    MTCL::Manager::init(groupName);
    std::atexit([]()-> void {MTCL::Manager::finalize();});

#if defined(ENABLE_MPI) || defined(DFF_MPI)
    if (dGroups::Instance()->baseProtocol == "MPI") {
      int myrank;
      MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
      dGroups::Instance()->setRunningGroupByRank(myrank);
    }
#endif

    // trig the mapping set if specified
    dGroups::Instance()->setThreadMapping();

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
	
} // namespace
#endif
