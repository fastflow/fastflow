#ifndef IR_HPP
#define IR_HPP
#include <ff/distributed/ff_network.hpp>
#include <ff/node.hpp>
#include <ff/all2all.hpp>
#include <ff/distributed/ff_ddefines.hpp>
#include <list>
#include <vector>
#include <map>
#include <numeric>

namespace ff {

class ff_IR_V2 {
    friend class dGroups;
public:

    // representation of the buckets, i.e., the structure of the nested all-to-all
    std::vector<std::vector<ff_node*>> bucketsDistribution;

    //std::map<ff_node*, std::pair<ff::svector<ff_node*>,ff::svector<ff_node*>>> feeders, consumers;

    std::unordered_map<ff_node*, IngressEgressChannels_t> channelsDictionary;

    // list of <groupname - endpoints> to connect to  (in MTCL an edpoint is just a string)
    std::vector<std::tuple<std::string, std::string, std::vector<ff_node*>>> destinationEndpoints;

    // listening endpoints
    std::string listeningEndpoint;

    // list of remote groups names that need to connect to this group. The size of this vector represent the number of connection to expect and number of EOS
    std::vector<std::string> ingressRemoteConnectionsGroupsName;

    // size of batch
    int batchSize = DEFAULT_BATCH_SIZE;

    int messageOTF = DEFAULT_MESSAGE_OTF;

    void print(){
        ff::cout << "******* BEGIN INTERMEDIATE REPRESENTATION ********\n";
        
        ff::cout << "\t (*) Listening endpoint: " << listeningEndpoint << std::endl;
        
        ff::cout << "\t (*) Connections in output (" << destinationEndpoints.size() << "):\n";
        for (auto& [name, endp, _] : destinationEndpoints)
            ff::cout << "\t\t - " <<  name << " (" << endp << ")\n";
        
        ff::cout << "\t (*) Connections in input ("<< ingressRemoteConnectionsGroupsName.size() <<"):\n";
        for (auto& name : ingressRemoteConnectionsGroupsName)
            ff::cout << "\t\t - " << name << std::endl;

        ff::cout << "\t (*) Group Structure (" << bucketsDistribution.size() << " levels):\n";
        for(size_t i = 0; i < bucketsDistribution.size(); i++){
            ff::cout << "\t\tLevel " << i << ": ";
            for(size_t j = 0; j < bucketsDistribution[i].size(); j++) 
                ff::cout << bucketsDistribution[i][j]->mioID_str << "[" << bucketsDistribution[i][j]->mioID << "]" << ((j < bucketsDistribution[i].size()-1) ? ", ": "");
            ff::cout << "\n";
        }

        ff::cout << "******** END INTERMEDIATE REPRESENTATION *********\n";
    }

};

} // namespace

#endif
