/*  
 *            
 *                    
 *   Node1-->Node2 --> Node3
 *
 *   G1: Node1   
 *   G2: Node2
 *   G3: Node3
 *                    
 */


#include <iostream>
#include <string>
#include <ff/dff.hpp>

using namespace ff;

struct Node1: ff_node_t<std::string>{
	Node1(std::string& filename):filename(filename) {}
    std::string* svc(std::string*){
		FILE* in = fopen(filename.c_str(), "r");
		if (in == NULL) {
			perror("fopen");
			std::cerr << "Cannot open " << filename << " for reading\n";
			return EOS;
		}
		std::string *out=new std::string;
		char buf[4096]={'\0'};
		while(1) {
			int r=0;
			if ((r=fread(&buf, 1, sizeof(buf), in)) == -1) {
				perror("fread");
				std::cerr << "Problem when reading data from " << filename << "\n";
			}
			
			if (r>0) *out += buf;
			if (feof(in)) break;
		}
		char hostname[256];
		gethostname(hostname, 256);
		std::cout << "G1[" << hostname << "]: file " << filename << " read from disk, now sending it to G2\n";		
		ff_send_out(out);
		fclose(in);
        return EOS;
    }
	const std::string &filename;
};

struct Node2: ff_node_t<std::string>{
	Node2(std::string& filename):filename(filename) {}
    std::string* svc(std::string* out){
		char hostname[256];
		gethostname(hostname, 256);		
		std::cout << "G2[" << hostname << "]: received file from G1\n";
		FILE* in = fopen(filename.c_str(), "r");
		if (in == NULL) {
			perror("fopen");
			std::cerr << "Cannot open " << filename << " for reading\n";
			return EOS;
		}
		char buf[4096]={'\0'};
		while(1) {
			int r=0;
			if ((r=fread(&buf, 1, sizeof(buf), in)) == -1) {
				perror("fread");
				std::cerr << "Problem when reading data from " << filename << "\n";
			}
			if (r>0) *out += buf;
			if (feof(in)) break;
		}
		std::cout << "G2[" << hostname << "]: file " << filename << " read, and appended, sending it to G3\n";		
		ff_send_out(out);
		fclose(in);
        return GO_ON;
    }
	const std::string &filename;
};

struct Node3: ff_node_t<std::string>{ 
	Node3(std::string& filename):filename(filename) {}
    std::string* svc(std::string* in){
		char hostname[256];
		gethostname(hostname, 256);
		std::cout << "G3[" << hostname << "]: received file from G2\n";
		FILE* out = fopen(filename.c_str(), "w+");
		if (out == NULL) {
			perror("fopen");
			std::cerr << "Cannot open " << filename << " for writing\n";
			return EOS;
		}
		if (fwrite(in->c_str(), 1, in->length(), out) != in->length()) {
			perror("fwrite");
			std::cerr << "Problem when writing data into " << filename << "\n";
		}
		std::cout << "G3[" << hostname << "]: output wrote into " << filename << "\n";
        return GO_ON;
    }
	const std::string &filename;
};

static void usage(const char *argv0) {
	std::cerr << "Use: " << argv0 << " -f infile1 -s infile2 -o outfile\n";
}


int main(int argc, char*argv[]){
	
    if (DFF_Init(argc, argv)<0 ) {
		error("DFF_Init\n");
		return -1;
	}
	extern char *optarg;
    const char optstr[]="f:s:o:";
	std::string file1="";
	std::string file2="";
	std::string file3="";
	long opt;
	while((opt=getopt(argc,argv,optstr)) != -1) {
		switch(opt) {
		case 'f': file1+=std::string(optarg); break;
		case 's': file2+=std::string(optarg); break;
		case 'o': file3+=std::string(optarg); break;
		default:
			std::cerr << "invalid command line option\n";
			usage(argv[0]);
			return -1;
		}
	}
	if (!file1.length() || !file2.length() || !file3.length()) {
		usage(argv[0]);
		return -1;
	}
	
    ff_pipeline pipe;
	Node1 n1(file1);
	Node2 n2(file2);
	Node3 n3(file3);
	pipe.add_stage(&n1);
	pipe.add_stage(&n2);
	pipe.add_stage(&n3);
	
    //----- defining the distributed groups ------

    n1.createGroup("G1");
    n2.createGroup("G2");
	n3.createGroup("G3");
	
    // -------------------------------------------
	
	if (pipe.run_and_wait_end()<0) {
		error("running the main pipe\n");
		return -1;
	}
	return 0;
}
