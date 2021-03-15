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

int DFF_Init(int &argc, char **&argv){
        int c;
        std::string groupName, configFile;

        while (1){
            static struct option long_options[] =
                {
                    {"DFF_GName", required_argument, 0, 'n'},
                    {"DFF_Config", required_argument, 0, 'c'},
                    {0, 0, 0, 0}
                };

            /* getopt_long stores the option index here. */
            int option_index = 0;
            c = getopt_long(argc, argv, "", long_options, &option_index);

            /* Detect the end of the options. */
            if (c == -1) break;

            switch (c){
            case 0:
                /* If this option set a flag, do nothing else now. */
                if (long_options[option_index].flag != 0)
                    break;
                printf("option %s", long_options[option_index].name);
                if (optarg)
                    printf(" with arg %s", optarg);
                printf("\n");
                break;

            case 'c':
                configFile = std::string(optarg);
                break;

            case 'n':
                groupName = std::string(optarg);
                break;

            case '?':
                /* getopt_long already printed an error message. */
                break;

            default:
                return -1;
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

        // if other arguments are passed, they are preserved!
        if (optind <= argc){
            optind--;
            char *exeName = argv[0];
            argc -= optind;
            argv += optind;
            argv[0] = exeName;
        }

        return 0;
    }

}

#endif
