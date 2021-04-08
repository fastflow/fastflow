/* 
 * FastFlow concurrent network:
 * 
 *           |--> MiNode 
 *  MoNode-->|           
 *           |--> MiNode 
 *  MoNode-->|           
 *           |--> MiNode 
 *
 * /<------- a2a ------>/
 * /<---- pipeMain ---->/
 */


#include <ff/dff.hpp>
#include <iostream>
#include <mutex>
#include <chrono>

std::mutex mtx;  // used only for pretty printing

float active_delay(int msecs) {
  // read current time
  float x = 1.25f;
  auto start = std::chrono::high_resolution_clock::now();
  auto end   = false;
  while(!end) {
    auto elapsed = std::chrono::high_resolution_clock::now() - start;
    auto msec = std::chrono::duration_cast<std::chrono::milliseconds>(elapsed).count();
    x *= sin(x) / atan(x) * tanh(x) * sqrt(x);
    if(msec>=msecs)
      end = true;
  }
  return x;
}

struct ExcType {
    int taskID;
    std::vector<char> data;
    ExcType(){}
    ExcType(int taskID, long length) : taskID(taskID), data(length, 'F') {}

    template<class Archive>
	void serialize(Archive & archive) {
		archive(taskID, data);
	}
};


struct MoNode : ff::ff_monode_t<int, ExcType>{
    int items, execTime;
    long dataLength;

    MoNode(int itemsToGenerate, int execTime, long dataLength): items(itemsToGenerate), execTime(execTime), dataLength(dataLength) {}

    ExcType* svc(int*){
       for(int i=0; i< items; i++){
            active_delay(this->execTime);
            ff_send_out(new ExcType(i, dataLength));
       }        
        return this->EOS;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[MoNode" << this->get_my_id() << "] Generated Items: " << items << std::endl;
    }
};

struct MiNode : ff::ff_minode_t<ExcType>{
    int processedItems = 0;
    int execTime;
    MiNode(int execTime) : execTime(execTime) {}

    ExcType* svc(ExcType* i){
        active_delay(this->execTime);
        ++processedItems;
        delete i;
        return this->GO_ON;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[MiNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};


int main(int argc, char*argv[]){
    
    DFF_Init(argc, argv);

    if (argc < 7){
        std::cout << "Usage: " << argv[0] << "#items #byteXitem #execTimeSource #execTimeSink #nw_sx #nw_dx"  << std::endl;
        return -1;
    }

    int items = atoi(argv[1]);
    long bytexItem = atol(argv[2]);
    int execTimeSource = atoi(argv[3]);
    int execTimeSink = atoi(argv[4]);
    int numWorkerSx = atoi(argv[5]);
    int numWorkerDx = atoi(argv[6]);

    ff_pipeline mainPipe;
    ff::ff_a2a a2a;

    mainPipe.add_stage(&a2a);

    std::vector<MoNode*> sxWorkers;
    std::vector<MiNode*> dxWorkers;

    for(int i = 0; i < numWorkerSx; i++)
        sxWorkers.push_back(new MoNode(ceil((double)items/numWorkerSx), execTimeSource, bytexItem));

    for(int i = 0; i < numWorkerDx; i++)
        dxWorkers.push_back(new MiNode(execTimeSink));

    a2a.add_firstset(sxWorkers);
    a2a.add_secondset(dxWorkers);

    //mainPipe.run_and_wait_end();

    auto g1 = a2a.createGroup("G1");
    auto g2 = a2a.createGroup("G2");

     for(int i = 0; i < numWorkerSx; i++) g1.out << sxWorkers[i];

     for(int i = 0; i < numWorkerDx; i++) g2.in << dxWorkers[i];

   if (mainPipe.run_and_wait_end()<0) {
		error("running mainPipe\n");
		return -1;
	}

}