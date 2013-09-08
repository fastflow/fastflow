/*
 * Picture:
 *
 *
 *     -----------               -------                ---------
 *    |           |             |       |              |         |
 *    | unpacker  |-------------|    f  |--------------| packer  |
 *    |           |  ON-DEMAND  |       |  FROM-ANY    |         |
 *     -----------    (COMM1)    -------    (COMM2)     ---------
 *                                
 */

// NOTES: 
//   1- by defining FFDISTRIBUTED to 0 (i.e. compiling with -DFFDISTRIBUTED=0) 
//      this test will be executed as a "standard" (non distributed) 3-stage 
//      ff_pipeline whose middle stage is a ff_farm
//
//   2- if either FFDISTRIBUTED and DNODE_FARM are defined, the middle dnode
//      stages are implemented as ff_farm. If DNODE_FARM is not defined the 
//      middle dnode stages are implemented as sequential dnodes.
//

#include <math.h>
#include <ffmacros.h>

#define STREAMLEN    2048
#define SIZE_IN      1024
// WORK_ON_SIZE should be less than or equal to SIZE_IN 
#define WORK_ON_SIZE 1024
#define SIZE_OUT     1024
#define K            1024

// input type
FFType_ptr_1(miotipo_in_t,  double*, A, SIZE_IN*sizeof(double));
// output type
FFType_ptr_1(miotipo_out_t, double*, B, SIZE_OUT*sizeof(double));

// ---------------- user's functions -------------------
void unpacker(miotipo_in_t *& res) {
    static int N=STREAMLEN;

    if (N == 0) res = NULL;
    else {
	double* A = new double[SIZE_IN];
	for(int i=0;i<SIZE_IN;++i) {
	    A[i] = 3.14/(double)N; 
	}
	res = new miotipo_in_t(A);
    }
    --N;
}
// main function 
void f(miotipo_in_t& in, miotipo_out_t *& res) {
    double* B = new double[SIZE_OUT];
    double t2;
    for(int i=0;i<WORK_ON_SIZE;++i) {
	t2 = in.A[i];
	for(int j=0;j<K;++j) t2 = sin(1/t2)*t2 + cos(1/t2)*t2;
	in.A[i] = t2;
    }
    if (SIZE_IN>= SIZE_OUT)
	memcpy(B,in.A,SIZE_OUT*sizeof(double));
    else
	memcpy(B,in.A,SIZE_IN*sizeof(double));

    res = new miotipo_out_t(B);
}
void packer(miotipo_out_t *& res) {    
  //printf("got result %.8f\n",res->B[0]);
}
// ---------------------------------------------------


// ------------------ dnode definitions --------------
PIPE3DEF(unpacker,f,packer, miotipo_in_t, miotipo_out_t);
// ---------------------------------------------------


int main(int argc, char* argv[]) {
    if (argc < 5) {
	std::cerr << "use:\n";
	std::cerr << " " << argv[0] << " hostid #dnode #workers host1:port host2:port\n\n";
	std::cerr << "   hostId    : the host identifier\n";
	std::cerr << "               -1       is the first stage\n";
	std::cerr << "               -2       is the last stage\n";
	std::cerr << "               -3       GWin  ( see also #worker )\n";
	std::cerr << "               -4       GWout ( see also #worker )\n";
	std::cerr << "               (0,n-1(  for middle stages\n\n";
	std::cerr << "   #dnode    : the number of dnodes\n\n";
	std::cerr << "   #workers  : the internal parallelism degree of each dnode\n";
	std::cerr << "               NOTE1: used only if DNODE_FARM is defined\n";
	std::cerr << "               NOTE2: In case of GWin/GWout this param specifies the 'virtual' hostId\n\n";
	std::cerr << "   host1:port: hostname/IP and TCP port of the host where\n";
	std::cerr << "               the unpacker node (first one) is executed or where\n";
	std::cerr << "               the GWin is executed\n\n";
	std::cerr << "   host2:port: hostname/IP and TCP port of the host where\n";
	std::cerr << "               the packer node (last one) is executed or where\n";
	std::cerr << "               the GWout is executed\n\n";


	return -1;
    }

    int      hostId     = atoi(argv[1]);
    int      nodedegree = atoi(argv[2]);
    int      pardegree  = atoi(argv[3]);   
    char*    address1   = argv[4];         // no check
    char*    address2   = argv[5];         // no check

#if defined(SEQUENTIAL)
    do {
      miotipo_in_t* in;
      miotipo_out_t* out;
      unpacker(in);
      if (in == NULL) break;
      f(*in,out);
      packer(out);
    }while(1);

#else
    FFd_init( ((hostId==-3 || hostId==-4) ? pardegree : hostId) );
    FFd_PIPE3(hostId, unpacker, f, packer, address1, address2,nodedegree, pardegree);
    FFd_end();
#endif
    return 0;
}
