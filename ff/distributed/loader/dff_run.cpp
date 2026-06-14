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
/* Author: 
 *   Nicolo' Tonci
 */

#include <iostream>
#include <fstream>
#include <sstream>
#include <chrono>
#include <thread>
#include <vector>
#include <algorithm>
#include <cstring>
#include <cerrno>
#include <cstdlib>
#include <csignal>
#include <unistd.h>
#include <sys/time.h>
#include <sys/wait.h>
#include <sys/param.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/socket.h>
#include <netdb.h>
#include <netinet/in.h>
#include <arpa/inet.h>



#include <cereal/cereal.hpp>
#include <cereal/archives/json.hpp>
#include <cereal/types/string.hpp>
#include <cereal/types/vector.hpp>


#include <filesystem>
namespace n_fs = std::filesystem;

#ifndef HOST_NAME_MAX
#define HOST_NAME_MAX 255
#endif

enum Proto {TCP = 1 , MPI};

Proto usedProtocol;
bool seeAll = false;
bool dryRun = false;
unsigned timeoutSec = 0;
constexpr auto TCP_LAUNCHER_POLL_INTERVAL = std::chrono::milliseconds(10);
std::vector<std::string> viewGroups;
char hostname[HOST_NAME_MAX];
std::string configFile("");
std::string executablePath;
std::vector<std::string> executableArgs;


static inline unsigned long getusec() {
    struct timeval tv;
    gettimeofday(&tv,NULL);
    return (unsigned long)(tv.tv_sec*1e6+tv.tv_usec);
}

static inline bool toBePrinted(std::string gName){
    return (seeAll || (find(viewGroups.begin(), viewGroups.end(), gName) != viewGroups.end()));
}

static inline std::string shellQuote(const std::string& s) {
    std::string out("'");
    for (char c : s) {
        if (c == '\'') out += "'\\''";
        else out += c;
    }
    out += "'";
    return out;
}

static inline std::vector<std::string> split (const std::string &s, char delim) {
    std::vector<std::string> result;
    std::stringstream ss(s);
    std::string item;

    while (getline (ss, item, delim))
        result.push_back (item);

    return result;
}

static inline void convertToIP(const char *host, char *ip) {
	struct addrinfo hints;
	struct addrinfo *result, *rp;
	memset(&hints, 0, sizeof(hints));
	hints.ai_family = AF_UNSPEC;    /* Allow IPv4 or IPv6 */
	hints.ai_socktype = SOCK_STREAM; /* Stream socket */
	hints.ai_flags = 0;
	hints.ai_protocol = IPPROTO_TCP;          /* Allow only TCP */
	if (getaddrinfo(host, NULL, NULL, &result) != 0) {
		perror("getaddrinfo");
		std::cerr << "FATAL ERROR\n";							
		return;
	}	
	for (rp = result; rp != NULL; rp = rp->ai_next) {
		struct sockaddr_in *h = (struct sockaddr_in *) rp->ai_addr;
		if (inet_ntop(AF_INET, &(h->sin_addr), ip, INET_ADDRSTRLEN) == NULL) {
			perror("inet_ntop");
			continue;
		}		
		free(result);
		return;
	}
	free(result);
	std::cerr << "FATAL ERROR\n";
}


struct G {
    std::string name, host, preCmd;
    int fd = -1;
    pid_t pid = -1;
    int exitCode = -1;
    bool signaled = false;

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

    // Build the exact command executed for this group. The executable and runtime
    // arguments are quoted individually; preCmd is kept as shell syntax so it
    // can provide environment setup before launching the group.
    std::string buildCommand() const {
        std::ostringstream cmd;
        if (!preCmd.empty())
            cmd << preCmd << " ";

        cmd << shellQuote(executablePath);
        for (const auto& arg : executableArgs)
            cmd << " " << shellQuote(arg);
        cmd << " " << shellQuote("--DFF_Config=" + configFile);
        cmd << " " << shellQuote("--DFF_GName=" + name);

        return cmd.str();
    }

    std::string printableCommand() const {
        std::string cmd = buildCommand();
        if (isRemote())
            return "ssh -T " + shellQuote(host) + " " + shellQuote(cmd);
        return cmd;
    }

    // Start the group process and capture both stdout and stderr through a pipe.
    // This keeps TCP mode independent from mpirun while still allowing the
    // launcher to report per-group failures.
    void run(){
        std::string command = buildCommand();
        std::cout << "Executing [" << name << "]: " << printableCommand() << std::endl;
        if (dryRun) return;

        int pipefd[2];
        if (pipe(pipefd) < 0) {
            perror("pipe");
            exit(EXIT_FAILURE);
        }

        pid = fork();
        if (pid < 0) {
            perror("fork");
            close(pipefd[0]);
            close(pipefd[1]);
            exit(EXIT_FAILURE);
        }

        if (pid == 0) {
            // Put the child in its own process group so timeout handling can
            // terminate the shell, ssh and the group process together.
            setpgid(0, 0);
            close(pipefd[0]);
            dup2(pipefd[1], STDOUT_FILENO);
            dup2(pipefd[1], STDERR_FILENO);
            close(pipefd[1]);

            // Remote execution still runs the same quoted group command, but
            // delegates process creation to ssh on the target host.
            if (isRemote()) {
                execlp("ssh", "ssh", "-T", host.c_str(), command.c_str(), (char*)nullptr);
            } else {
                execlp("/bin/sh", "sh", "-lc", command.c_str(), (char*)nullptr);
            }
            perror("exec");
            _exit(127);
        }

        close(pipefd[1]);
        fd = pipefd[0];

        int flags = fcntl(fd, F_GETFL, 0);
        if (flags >= 0)
            fcntl(fd, F_SETFL, flags | O_NONBLOCK);
    }

    // Treat loopback, the local hostname and DFF_RUN_HOSTNAME as local.
    // Other hostnames are resolved and compared against this host IP.
    bool isRemote() const {
		if (!host.compare("127.0.0.1") || !host.compare("localhost") || !host.compare(hostname) || !(std::getenv("DFF_RUN_HOSTNAME") && host.compare(std::getenv("DFF_RUN_HOSTNAME"))))
			return false;
		
		char ip1[INET_ADDRSTRLEN];
		char ip2[INET_ADDRSTRLEN];
		convertToIP(host.c_str(), ip1);
		convertToIP(hostname, ip2);
		if (strncmp(ip1,ip2,INET_ADDRSTRLEN)==0) return false;
		return true;  // remote
	}

    bool isRunning() const {
        return pid > 0;
    }

    void closeOutput() {
        if (fd >= 0) {
            close(fd);
            fd = -1;
        }
    }

    void terminate() {
        // Give the group a short graceful-stop window before forcing cleanup.
        if (pid <= 0) return;
        kill(-pid, SIGTERM);
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
        if (pid > 0)
            kill(-pid, SIGKILL);
    }
};

bool allTerminated(const std::vector<G>& groups){
    for (const G& g: groups)
        if (g.isRunning())
            return false;
    return true;
}

static inline void usage(char* progname) {
	std::cout << "\nUSAGE: " <<  progname << " [Options] -f <configFile> <cmd> \n"
			  << "Options: \n"
			  << "\t -v <g1>,...,<g2> \t Prints the output of the specified groups\n"
			  << "\t -V               \t Print the output of all groups\n"
			  << "\t -p \"TCP|MPI\"   \t Force communication protocol\n";
	std::cout << "\n";
		
}

std::string generateRankFile(std::vector<G>& parsedGroups){
    std::string name = "/tmp/dffRankfile" + std::to_string(getpid());

    std::ofstream tmpFile(name, std::ofstream::out);
    
    for(size_t i = 0; i < parsedGroups.size(); i++)
        tmpFile << "rank " << i << "=+n" << i << " slot=0:*\n";  // TODO: to use the "threadMapping" attribute

    tmpFile.close();
    return name;
}

std::string generateHostFile(std::vector<G>& parsedGroups){
    std::string name = "/tmp/dffHostFile" + std::to_string(getpid());

    std::ofstream tmpFile(name, std::ofstream::out);
    
    for(size_t i = 0; i < parsedGroups.size(); i++)
        tmpFile << parsedGroups[i].host << " slots=1\n"; 

    tmpFile.close();
    return name;
}

int main(int argc, char** argv) {

    if (argc == 1 ||
		strcmp(argv[1], "--help") == 0 || strcmp(argv[1], "-help") == 0 || strcmp(argv[1], "-h") == 0){
		usage(argv[0]);
        exit(EXIT_SUCCESS);
    }

    if (gethostname(hostname, HOST_NAME_MAX) != 0) {
		perror("gethostname");
		exit(EXIT_FAILURE);
	}

	int optind=0;
	for(int i=1;i<argc;++i) {
		if (argv[i][0]=='-') {
			switch(argv[i][1]) {
            case 'p' : {
                if (argv[i+1] == NULL) {
                    std::cerr << "-p require a protocol\n";
                    usage(argv[0]);
					exit(EXIT_FAILURE);
                }
                std::string forcedProtocol = std::string(argv[++i]);
                if (forcedProtocol == "MPI")      usedProtocol = Proto::MPI;
                else if (forcedProtocol == "TCP") usedProtocol = Proto::TCP;
                else {
                    std::cerr << "-p require a valid protocol (TCP or MPI)\n";
					exit(EXIT_FAILURE);
                }
            } break;
			case 'f': {
				if (argv[i+1] == NULL) {
					std::cerr << "-f requires a file name\n";
					usage(argv[0]);
					exit(EXIT_FAILURE);
				}
				configFile = n_fs::absolute(n_fs::path(argv[++i])).string();
			} break;
			case 'V': {
				seeAll=true;
			} break;
            case 'n': {
                dryRun = true;
            } break;
            case 't': {
                if (argv[i+1] == NULL) {
                    std::cerr << "-t requires a timeout in seconds\n";
                    usage(argv[0]);
                    exit(EXIT_FAILURE);
                }
                timeoutSec = std::stoul(argv[++i]);
            } break;
			case 'v': {
				if (argv[i+1] == NULL) {
					std::cerr << "-v requires at list one argument\n";
					usage(argv[0]);
					exit(EXIT_FAILURE);
				}
				viewGroups = split(argv[++i], ',');
			} break;
			}
		} else { optind=i; break;}
	}

	if (configFile == "") {
		std::cerr << "ERROR: Missing config file for the loader\n";
		usage(argv[0]);
		exit(EXIT_FAILURE);
	}

    if (optind <= 0 || optind >= argc) {
        std::cerr << "ERROR: Missing executable command\n";
        usage(argv[0]);
        exit(EXIT_FAILURE);
    }

    executablePath = n_fs::absolute(n_fs::path(argv[optind])).string();

	if (!n_fs::exists(executablePath)) {
		std::cerr << "ERROR: Unable to find the executable file (we found as executable \'" << argv[optind] << "\')\n";
		exit(EXIT_FAILURE);
	}
		
    for (int index = optind+1 ; index < argc; index++) {
        executableArgs.push_back(argv[index]);
	}
	
    std::ifstream is(configFile);

    if (!is){
        std::cerr << "Unable to open configuration file for the program!" << std::endl;
        return -1;
    }

    std::vector<G> parsedGroups;

    try {
        cereal::JSONInputArchive ar(is);

        /*get the protocol to be used from the configuration file if it was not forced by the command line*/
        if (!usedProtocol)
            try {
                std::string tmpProtocol;
                ar(cereal::make_nvp("protocol", tmpProtocol));
                if (tmpProtocol == "MPI")
                    usedProtocol = Proto::MPI;
                else 
                    usedProtocol = Proto::TCP;
            } catch (cereal::Exception&) {
                ar.setNextName(nullptr);
                /*if the protocol is not specified we assume TCP*/
                usedProtocol = Proto::TCP;
            }

        //parse all the groups in the configuration file
        ar(cereal::make_nvp("groups", parsedGroups));
    } catch (const cereal::Exception& e){
        std::cerr << "Error parsing the JSON config file. Check syntax and structure of  the file and retry!" << std::endl;
        exit(EXIT_FAILURE);
    }

    #ifdef DEBUG
        for(auto& g : parsedGroups)
            std::cout << "Group: " << g.name << " on host " << g.host << std::endl;
    #endif

    if (usedProtocol == Proto::TCP){
        auto Tstart = getusec();
        for (G& g : parsedGroups)
            g.run();

        if (dryRun)
            return 0;

        bool timedOut = false;
        
        while(!allTerminated(parsedGroups)){
            if (timeoutSec > 0 && ((getusec() - Tstart) / 1000000) >= timeoutSec) {
                // Timeout is enforced by the launcher so a hung TCP group
                // cannot keep the whole application alive indefinitely.
                timedOut = true;
                for (G& g : parsedGroups)
                    if (g.isRunning())
                        g.terminate();
            }

            for(G& g : parsedGroups){
                if (g.isRunning()){
                    char buff[1024] = { 0 };
                    
                    // Pipes are non-blocking: no available output should not
                    // delay polling the other still-running groups.
                    ssize_t result = g.fd >= 0 ? read(g.fd, buff, sizeof(buff)) : 0;
                    if (result == -1){
                        if (errno == EAGAIN || errno == EWOULDBLOCK)
                            continue;
                        g.closeOutput();
                    } else if (result > 0){
                        if (toBePrinted(g.name))
                            std::cout << buff;
                    } else {
                        g.closeOutput();
                    }

                    // Reap completed children without blocking. Drain any
                    // remaining pipe data before recording the final status.
                    int status = 0;
                    pid_t r = waitpid(g.pid, &status, WNOHANG);
                    if (r == g.pid) {
                        while (g.fd >= 0) {
                            result = read(g.fd, buff, sizeof(buff));
                            if (result > 0) {
                                if (toBePrinted(g.name))
                                    std::cout << buff;
                            } else break;
                        }
                        g.closeOutput();

                        if (WIFEXITED(status)) {
                            g.exitCode = WEXITSTATUS(status);
                            if (g.exitCode != 0)
                                std::cout << "[" << g.name << "][ERR] returned code " << g.exitCode << std::endl;
                        } else if (WIFSIGNALED(status)) {
                            g.signaled = true;
                            g.exitCode = 128 + WTERMSIG(status);
                            std::cout << "[" << g.name << "][ERR] killed by signal " << WTERMSIG(status) << std::endl;
                        }
                        g.pid = -1;
                    }
                }
            }

            std::this_thread::sleep_for(TCP_LAUNCHER_POLL_INTERVAL);
        }
        std::cout << "Elapsed time: " << (getusec()-(Tstart))/1000 << " ms" << std::endl;

        if (timedOut) {
            std::cerr << "ERROR: timeout after " << timeoutSec << " seconds\n";
            return EXIT_FAILURE;
        }

        for (const G& g : parsedGroups)
            if (g.exitCode != 0)
                return EXIT_FAILURE;
    }

    if (usedProtocol == Proto::MPI){
        std::string rankFile = generateRankFile(parsedGroups);
        std::string hostFile = generateHostFile(parsedGroups);
        std::cout << "RankFile: " << rankFile << std::endl;
        // invoke mpirun using the just created rankfile

        char command[350];
     
        std::ostringstream exe;
        exe << shellQuote(executablePath);
        for (const auto& arg : executableArgs)
            exe << " " << shellQuote(arg);

        sprintf(command, "mpirun --hostfile %s -np %lu --rankfile %s %s --DFF_Config=%s", hostFile.c_str(), parsedGroups.size(), rankFile.c_str(), exe.str().c_str(), configFile.c_str());

		std::cout << "mpicommand: " << command << "\n";
		
        FILE *fp;
        char buff[1024];
        fp = popen(command, "r");
        if (fp == NULL) {
            printf("Failed to run command\n" );
            exit(1);
        }

        /* Read the output a line at a time - output it. */
        while (fgets(buff, sizeof(buff), fp) != NULL) {
            std::cout << buff;
        }

        pclose(fp);

        std::remove(rankFile.c_str());
        std::remove(hostFile.c_str());
    }
    
    
    return 0;
}
