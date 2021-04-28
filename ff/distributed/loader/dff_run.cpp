#include <iostream>
#include <fstream>
#include <chrono>
#include <thread>
#include <unistd.h>
#include <sys/time.h>
#include <sys/wait.h>


#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>

#if(defined(_MSC_VER) or (defined(__GNUC__) and (7 <= __GNUC_MAJOR__)))
    #include <filesystem>
    namespace n_fs = std::filesystem;
#else
    #include <experimental/filesystem>
    namespace n_fs = std::experimental::filesystem;    
#endif


static inline unsigned long getusec() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)(tv.tv_sec*1e6+tv.tv_usec);
}

char hostname[100];
std::string configFile("");
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
    std::string name, host, preCmd;
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

        try {
            ar(cereal::make_nvp("preCmd", preCmd)); 
        } catch (cereal::Exception&) {
            ar.setNextName(nullptr);
        }
    }

    void run(){
        char b[350]; // ssh -t
        sprintf(b, " %s %s %s %s --DFF_Config=%s --DFF_GName=%s", (isRemote() ? "ssh -t " : ""), (isRemote() ? host.c_str() : "") , this->preCmd.c_str(),  executable.c_str(), configFile.c_str(), this->name.c_str());
       std::cout << "Executing the following command: " << b << std::endl;
        fd = popen(b, "r");

        if (fd == NULL) {
            printf("Failed to run command\n" );
            exit(1);
        }
    }

    bool isRemote(){return !(!host.compare("127.0.0.1") || !host.compare("localhost") || !host.compare(hostname));}


};

bool allTerminated(std::vector<G>& groups){
    for (G& g: groups)
        if (g.fd != nullptr)
            return false;
    return true;
}

static inline void usage(char* progname) {
	std::cout << "\nUSAGE: " <<  progname << " [Options] -f <configFile> <cmd> \n"
			  << "Options: \n"
			  << "\t -v <g1>,...,<g2> \t Prints the output of the specified groups\n"
			  << "\t -V               \t Print the output of all groups\n";
		
}

int main(int argc, char** argv) {

    if (strcmp(argv[0], "--help") == 0 || strcmp(argv[0], "-help") == 0 || strcmp(argv[0], "-h") == 0){
		usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    // get the hostname
    gethostname(hostname, 100);

    std::vector<std::string> viewGroups;
    bool seeAll = false;
	int optind=0;
	for(int i=1;i<argc;++i) {
		if (argv[i][0]=='-') {
			switch(argv[i][1]) {
			case 'f': {
				if (argv[i+1] == NULL) {
					std::cerr << "-f requires a file name\n";
					usage(argv[0]);
					exit(EXIT_FAILURE);
				}
				++i;
				configFile = std::string(argv[i]);
			} break;
			case 'V': {
				seeAll=true;
			} break;
			case 'v': {
				if (argv[i+1] == NULL) {
					std::cerr << "-v requires at list one argument\n";
					usage(argv[0]);
					exit(EXIT_FAILURE);
				}
				viewGroups = split(argv[i+1], ',');
				i+=viewGroups.size();
			} break;
			}
		} else { optind=i; break;}
	}

	if (configFile == "") {
		std::cerr << "ERROR: Missing config file for the loader\n";
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}
	if (!n_fs::exists(std::string(argv[optind]))) {
		std::cerr << "ERROR: Unable to find the executable file (we found as executable \'" << argv[optind] << "\')\n";
		exit(EXIT_FAILURE);
	}	
		
    for (int index = optind; index < argc; index++) {
        executable += std::string(argv[index]) + " ";
	}
	
    std::ifstream is(configFile);

    if (!is){
        std::cerr << "Unable to open configuration file for the program!" << std::endl;
        return -1;
    }

    std::vector<G> parsedGroups;

    try {
        cereal::JSONInputArchive ar(is);
        ar(cereal::make_nvp("groups", parsedGroups));
    } catch (const cereal::Exception& e){
        std::cerr << "Error parsing the JSON config file. Check syntax and structure of  the file and retry!" << std::endl;
        exit(EXIT_FAILURE);
    }

    #ifdef DEBUG
        for(auto& g : parsedGroups)
            std::cout << "Group: " << g.name << " on host " << g.host << std::endl;
    #endif

    auto Tstart = getusec();

    for (G& g : parsedGroups)
        g.run();
    
    while(!allTerminated(parsedGroups)){
        for(G& g : parsedGroups){
            if (g.fd != nullptr){
                char buff[1024];
                char* result = fgets(buff, sizeof(buff), g.fd);
                if (result == NULL){
                    int code = pclose(g.fd);
                    if (WEXITSTATUS(code) != 0)
                        std::cout << "[" << g.name << "][ERR] Report an return code: " << WEXITSTATUS(code) << std::endl;
                    g.fd = nullptr;
                } else {
                    if (seeAll || find(viewGroups.begin(), viewGroups.end(), g.name) != viewGroups.end())
                        std::cout << "[" << g.name << "]" << buff;
                }
            }
        }

    std::this_thread::sleep_for(std::chrono::milliseconds(15));
    }
    
    std::cout << "Elapsed time: " << (getusec()-(Tstart))/1000 << " ms" << std::endl;


    return 0;
}
