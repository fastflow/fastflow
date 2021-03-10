#include <ff/dff.hpp>
#include <iostream>
#include <mutex>

#define ITEMS 100
std::mutex mtx;  // used only for pretty printing

using namespace ff;

struct Source : ff::ff_monode_t<int>{
    int* svc(int* i){
        for(int i=0; i< ITEMS; i++)
            ff_send_out(new int(i));
        
        return this->EOS;
    }
};

struct MoNode : ff::ff_monode_t<int>{
    int processedItems = 0;
    int* svc(int* i){
        ++processedItems;
        return i;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[SxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};

struct MiNode : ff::ff_minode_t<int>{
    int processedItems = 0;
    int* svc(int* i){
        ++processedItems;
        return i;
    }

    void svc_end(){
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "[DxNode" << this->get_my_id() << "] Processed Items: " << processedItems << std::endl;
    }
};

struct Sink : ff::ff_minode_t<int>{
    int sum = 0;
    int* svc(int* i){
        sum += *i;
        delete i;
        return this->GO_ON;
    }

    void svc_end() {
        int local_sum = 0;
        for(int i = 0; i < ITEMS; i++) local_sum += i;
        const std::lock_guard<std::mutex> lock(mtx);
        std::cout << "Sum: " << sum << " (Expected: " << local_sum << ")" << std::endl;
    }
};


int main(int argc, char*argv[]){
    
    DFF_Init(argc, argv);

    ff_pipeline mainPipe;
    ff::ff_a2a a2a;
    Source s;
    Sink sink;
    ff_Pipe sp(s, a2a);
    ff_Pipe sinkp(sink);
    mainPipe.add_stage(&sp);
    mainPipe.add_stage(&sinkp);

    MoNode sx1, sx2, sx3;
    MiNode dx1, dx2, dx3;

    a2a.add_firstset<MoNode>({&sx1, &sx2, &sx3});
    a2a.add_secondset<MiNode>({&dx1, &dx2, &dx3});


    
    dGroup g1 = sp.createGroup("G1");
    dGroup g3 = sinkp.createGroup("G3");


    g1.out << &dx1 << &dx2 << &dx3;
    g3.in << &sink;
    
    mainPipe.run_and_wait_end();
    
    return 0;
}