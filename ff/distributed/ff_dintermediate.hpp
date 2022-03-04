#ifndef IR_HPP
#define IR_HPP
#include <ff/distributed/ff_network.hpp>
#include <ff/node.hpp>
#include <ff/all2all.hpp>
#include <list>
#include <vector>
#include <map>

namespace ff {

class ff_IR {
    friend class dGroups;
protected:
    // set to true if the group contains the whole parent building block
    bool wholeParent = false;
    void computeCoverage(){
        if (!parentBB->isAll2All()) return;
        ff_a2a* a2a = reinterpret_cast<ff_a2a*>(parentBB);
        coverageL = coverageR = true;
        for(ff_node* n : a2a->getFirstSet())
            if (!L.contains(n)) {coverageL = false; break;}
        for(ff_node* n : a2a->getSecondSet())
            if (!R.contains(n)) {coverageR = false; break;}
    }

    void buildIndexes(){

        if (!L.empty()){
            ff::svector<ff_node*> parentInputs;
            parentBB->get_in_nodes(parentInputs);

            ff::svector<ff_node*> LOutputs;
            if (!parentBB->isAll2All() || wholeParent) parentBB->get_out_nodes(LOutputs);
            else for(ff_node* n : reinterpret_cast<ff_a2a*>(parentBB)->getFirstSet()) n->get_out_nodes(LOutputs);
            
            for(ff_node* n : L){
                ff::svector<ff_node*> bbInputs; n->get_in_nodes(bbInputs);     
                for(ff_node* bbInput : bbInputs)
                    inputL.push_back(std::find(parentInputs.begin(), parentInputs.end(), bbInput) - parentInputs.begin());

                ff::svector<ff_node*> bbOutputs; n->get_out_nodes(bbOutputs);
                for(ff_node* bbOutput : bbOutputs)
                    outputL.push_back(std::find(LOutputs.begin(), LOutputs.end(), bbOutput) - LOutputs.begin());
            }
        }

        if (!R.empty() && parentBB->isAll2All() && !wholeParent){
            ff::svector<ff_node*> RInputs;
            for(ff_node* n : reinterpret_cast<ff_a2a*>(parentBB)->getSecondSet()) n->get_in_nodes(RInputs);

            ff::svector<ff_node*> parentOutputs;
            parentBB->get_out_nodes(parentOutputs);
            
            for(ff_node* n : R){
                ff::svector<ff_node*> bbInputs; n->get_in_nodes(bbInputs);
                for(ff_node* bbInput : bbInputs)
                    inputR.push_back(std::find(RInputs.begin(), RInputs.end(), bbInput) - RInputs.begin());

                ff::svector<ff_node*> bbOutputs; n->get_out_nodes(bbOutputs);
                for(ff_node* bbOutput : bbOutputs)
                    outputR.push_back(std::find(parentOutputs.begin(), parentOutputs.end(), bbOutput) - parentOutputs.begin());
            }
        }

    }

    
public:
    std::set<ff_node*> L, R;
    bool coverageL = false, coverageR = false;
    bool isSource = false, isSink = false;
    bool hasReceiver = false, hasSender = false;

    ff_node* parentBB;

    ff_endpoint listenEndpoint;
    std::vector<ff_endpoint> destinationEndpoints;
    std::set<std::string> otherGroupsFromSameParentBB;
    size_t expectedEOS = 0;

    // liste degli index dei nodi input/output nel builiding block in the shared memory context. The first list: inputL will become the rouitng table
    std::vector<int> inputL, outputL, inputR, outputR;

    std::map<std::string, std::pair<std::vector<int>, bool>> routingTable;
    
    // TODO: implmentare l'assegnamento di questi campi
    int leftTotalOuputs;
    int rightTotalInputs;

    bool isVertical(){return (L.empty() + R.empty()) == 1;}

    bool hasLeftChildren() {return !L.empty();}
    bool hasRightChildren() {return !R.empty();}

    void insertInList(std::pair<ff_node*, SetEnum> bb, bool _wholeParent = false){
        wholeParent = _wholeParent;
        switch(bb.second){
            case SetEnum::L: L.insert(bb.first); return;
            case SetEnum::R: R.insert(bb.first); return;
        }
    }

    std::vector<int> getInputIndexes(bool internal){
        if (internal && !R.empty()) return inputR;
        return inputL;
    }

    void print(){
        std::cout << "###### BEGIN GROUP ######\n";
        std::cout << "Group Orientation: " << (isVertical() ? "vertical" : "horizontal") << std::endl;
        std::cout << std::boolalpha << "Source group: " << isSource << std::endl;
        std::cout << std::boolalpha << "Sink group: " << isSink << std::endl;
        std::cout << std::boolalpha << "Coverage Left: " << coverageL << std::endl;
        std::cout << std::boolalpha << "Coverage Right: " << coverageR << std::endl << std::endl;

        std::cout << std::boolalpha << "Has Receiver: " << hasReceiver << std::endl;
        std::cout << "Expected input connections: " << expectedEOS << std::endl;
        std::cout << "Listen endpoint: " << listenEndpoint.address << ":" << listenEndpoint.port << std::endl << std::endl;

        std::cout << std::boolalpha << "Has Sender: " << hasSender << std::endl;
        std::cout << "Destination endpoints: " << std::endl;
        for(ff_endpoint& e : destinationEndpoints)
            std::cout << "\t* " << e.groupName << "\t[[" << e.address << ":" << e.port << "]]" << std::endl;
        
        std::cout << "\n\nIndex Input Left: ";
        for(int i : inputL) std::cout << i << " ";
        std::cout << "\n";

        std::cout << "Index Output Left: ";
        for(int i : outputL) std::cout << i << " ";
        std::cout << "\n";

        std::cout << "Index Input Right: ";
        for(int i : inputR) std::cout << i << " ";
        std::cout << "\n";

        std::cout << "Index Output Right: ";
        for(int i : outputR) std::cout << i << " ";
        std::cout << "\n";
        
        std::cout << "######  END GROUP  ######\n";
    }


};


}

#endif