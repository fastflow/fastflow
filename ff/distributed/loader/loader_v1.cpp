#include <iostream>
#include <fstream>
#include <unistd.h>
#include <../external/cereal/cereal.hpp>
#include <../external/cereal/archives/json.hpp>
#include <../external/cereal/types/string.hpp>
#include <../external/cereal/types/vector.hpp>

std::string configFile;
std::string executable;

inline std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss (s);
    std::string item;

    while (getline (ss, item, delim))
        result.push_back (item);

    return result;
}

struct G {
    std::string name, host;
    FILE* fd = nullptr;

    template <class Archive>
    void load( Archive & ar ){
        ar(cereal::make_nvp("name", name));
        
        try {
            std::string endpoint;
            ar(cereal::make_nvp("endpoint", endpoint)); std::vector endp(split(endpoint, ':'));
            host = endp[0]; //port = std::stoi(endp[1]);
        } catch (cereal::Exception&) {
            host = "127.0.0.1"; // set the host to localhost if not found in config file!
            ar.setNextName(nullptr);
            }
        
        std::cout << "Host: " << host << std::endl;
    }

    void run(){
        char b[100]; // ssh -t
        sprintf(b, " %s %s %s --DFF_Config=%s --DFF_GName=%s", (isRemote() ? "ssh -t " : ""), (isRemote() ? host.c_str() : "") , executable.c_str(), configFile.c_str(), this->name.c_str());
       std::cout << "Executing the following string: " << b << std::endl;
        fd = popen(b, "r");

        if (fd == NULL) {
            printf("Failed to run command\n" );
            exit(1);
        }
    }

    bool isRemote(){return !(!host.compare("127.0.0.1") || !host.compare("localhost"));}


};

bool allTerminated(std::vector<G>& groups){
    for (G& g: groups)
        if (g.fd != nullptr)
            return false;
    return true;
}

int main(int argc, char** argv) {

    std::vector<std::string> viewGroups;
    
    int c;
    while ((c = getopt (argc, argv, ":f:v:")) != -1)
        switch (c){
            case 'f':
                configFile = std::string(optarg);
                break;
            case 'v':
                viewGroups = split(optarg, ',');
                break;
            case ':':
                if (optopt == 'v'){
                    // see all
                }
                break;
            case '?':
                if (optopt == 'f')
                fprintf (stderr, "Option -%c requires an argument.\n", optopt);
                else if (isprint (optopt))
                fprintf (stderr, "Unknown option `-%c'.\n", optopt);
                else
                fprintf (stderr, "Unknown option character `\\x%x'.\n", optopt);
                return 1;
            default:
                abort();
        }
    
    for (int index = optind; index < argc; index++)
        executable += std::string(argv[index]) + " ";
    
    std::ifstream is(configFile);

    if (!is) throw std::runtime_error("Unable to open configuration file for the program!");
        
    cereal::JSONInputArchive ar(is);

    std::vector<G> parsedGroups;

    try {
        ar(cereal::make_nvp("groups", parsedGroups));
    } catch (const cereal::Exception& e){
        std::cerr <<  e.what();
        exit(EXIT_FAILURE);
    }

    #ifdef DEBUG
        for(auto& g : parsedGroups)
            std::cout << "Group: " << g.name << " on host " << g.host << std::endl;
    #endif

    for (G& g : parsedGroups)
        g.run();

    
    while(!allTerminated(parsedGroups)){
        for(G& g : parsedGroups){
            if (g.fd != nullptr){
                char buff[1024];
                char* result = fgets(buff, sizeof(buff), g.fd);
                if (result == NULL){
                    pclose(g.fd); g.fd = nullptr;
                } else {
                    if (find(viewGroups.begin(), viewGroups.end(), g.name) != viewGroups.end())
                        std::cout << "[" << g.name << "]" << buff;
                }
            }
        }

        sleep(0.20);
    }
    
    std::cout << "Everything terminated correctly" << std::endl;

    return 0;
}