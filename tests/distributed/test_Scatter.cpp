/* 
 * FastFlow concurrent network:
 * 
 *          |--> RNode 
 *  LNode-->|           
 *          |--> RNode 
 *  LNode-->|           
 *          |--> RNode 
 *  LNode-->|
 *          |--> RNode
 *           
 * /<------- a2a ------>/
 *
 * distributed group names: 
 *  Si: LNode_i
 *  Di: RNode_i
 *
 * --------------------------------------
 *
 * How it works:
 * Let's suppose size=10, nLeft=3, nRight=5. 
 * Each R-Worker has to receive a partition of size 2 (size/nRight).
 * The number of chunks (i.e., the n. of partitions to be sent as seen by all L-Workers) is nRight.
 * The L-Worker with id 0 should send 2 chunks, chunk_0 to the L-Worker_0, chunk_1 to the L-Worker_1.
 * The L-Worker with id 1 should send 2 chunks, chunk_2 to the L-Worker_2, chunk_3 to the L-Worker_3.
 * The L-Worker with id 2 should send 1 chunk, chunk_4 to the L-Worker_4.
 * 
 */


// TODO: MPI con 32GB bomba (dimensione dei messaggi)
//       MPI con 16GB ok ma va molto peggio di TCP.


// running the tests with limited buffer capacity
#define FF_BOUNDED_BUFFER
#define DEFAULT_BUFFER_CAPACITY 128

#include <sys/mman.h>

#include <ff/dff.hpp>
#include <iostream>
#include <mutex>
#include <chrono>


using namespace ff;

using DataType = char;

size_t PARTITION_SIZE = 0;

template<typename T>
void serializefreetask(T *o, DataType* input) {
}
template<typename Buffer>
bool serialize(Buffer&b, DataType* input){
	b = {input, PARTITION_SIZE};
	return false;  // the data is not copied
}
template<typename Buffer>
bool deserialize(const Buffer&b, DataType* p){
	assert(b.second == PARTITION_SIZE);
	return false;
}
template<typename Buffer>
void deserializealloctask(const Buffer& b, DataType*& p) {
	assert(b.second == PARTITION_SIZE);	
	p = new (b.first) DataType[PARTITION_SIZE];
};


struct LNode : ff::ff_monode_t<DataType>{

    LNode(char* data, size_t size, long nLeft,long id):
		data(data), size(size), nLeft(nLeft), myid(id) {}

    DataType* svc(DataType*) {
		int  nout = get_num_outchannels();  // the number of R-Workers
		//long myid = get_my_id();         TODO: get_my_id() should give the global id! <--------------
		assert((size % nout)==0);           // this is a constraint that could be removed
		
		size_t partitionsize    = size/nout;           // the size of each partition
		size_t totalchunks      = nout;                // these are the total n. of partitions as observed by the L-Workers (size/partitionsize)
		size_t chunksforeachone = totalchunks / nLeft; // base n. of partition assigned (to be sent) to each L-Workers
		long   r                = totalchunks % nLeft;
		size_t mychunks         = chunksforeachone; 
		if (myid < r)     mychunks++;                  // the actual n. of partition assigned to the current LNode
		
		size_t startchunk = 0;
		char*  pstart     = data;
		long   startdest  = 0;
		if (myid != 0) {
			startchunk  = myid*chunksforeachone;
			if (r) 	startchunk += ((r<myid)?r:myid);  // starting chunk for the current LNode
				
			pstart     += partitionsize*startchunk;   // starting address
			startdest   = startchunk;                 // first R-Worker target
		}
		
		for(size_t i=0; i< mychunks; i++){
			assert((int)(startdest+i)<nout);
			//ff::cout << "[LNode" << myid << "] sending to " << startdest+i << "\n";
			long* p = reinterpret_cast<long*>(pstart);
			*p = (startdest+i + 1);
			ff_send_out_to(pstart, startdest+i);
			pstart += partitionsize;
		}
		return this->EOS;
    }
	char*   data;
	size_t  size;
	long    nLeft;
	long    myid;
};

struct RNode : ff::ff_minode_t<DataType>{
    int processedItems = 0;
	long myid =-1;

	RNode(long id):myid(id) {}
	
    DataType* svc(DataType* in){

		long* p = reinterpret_cast<long*>(in);
		if (*p != (myid+1)) {
			ff::cout << "ERROR: expected " << myid+1 << " reiceived " << *p << "\n";
		}
		++processedItems;
		delete [] in;
		return GO_ON;
    }
	
    void svc_end(){
		if (processedItems != 1) {
			std::cerr << "ERROR, something went wrong; " << get_my_id() << " received more than 1 partition!\n";
			abort();
		}
    }
};

int main(int argc, char*argv[]){
    
    if (DFF_Init(argc, argv) != 0) {
		error("DFF_Init\n");
		return -1;
	}

    if (argc != 4){
        std::cout << "Usage: " << argv[0] << " nLeft nRight size"  << std::endl;
        return -1;
    }
    long nLeft   = std::stol(argv[1]);
	long nRight  = std::stol(argv[2]);
	long size    = std::stol(argv[3]);

	if (size % nRight) {
		std::cerr << "ERROR, size should be equally divisible among the R-Workers!\n";
		return -1;
	}

	// sets the global variable 
	PARTITION_SIZE= size / nRight;
	
	char* m = (char*)mmap(NULL, size, PROT_READ|PROT_WRITE, MAP_ANONYMOUS|MAP_PRIVATE,0,0);
	if (m == MAP_FAILED) {
		perror("mmap");
		return -1;
	}

    ff::ff_a2a a2a;

    std::vector<LNode*> sxWorkers;
    std::vector<RNode*> dxWorkers;

    for(long i = 0; i < nLeft; i++) {
		LNode* n = new LNode(m,size,nLeft,i);
        sxWorkers.push_back(n);
		a2a.createGroup(std::string("S")+std::to_string(i)) << n;		
	}

    for(long i = 0; i < nRight; i++) {
		RNode* n = new RNode(i);
        dxWorkers.push_back(n);
		a2a.createGroup(std::string("D")+std::to_string(i)) << n;
	}

    a2a.add_firstset(sxWorkers, 0, true);
    a2a.add_secondset(dxWorkers, true);

	auto t0=getusec();
    if (a2a.run_and_wait_end()<0) {
      error("running mainPipe\n");
      return -1;
    }	
	ff::cout << "Total Time (ms) = " << (getusec()-t0)/1000.0 << "\n";

	munmap(m, size);
	return 0;
}
