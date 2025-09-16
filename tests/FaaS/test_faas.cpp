#include <ff/ff_faas.hpp>
#include "ff/FaaS/connectors/Serverledge_connector.hpp"
#include <iostream>

int NUM_TASK = 3; // Numero di task da generare
#define F_NAME "testfaasfunction"
#define F_NAME2 "testfaasfunction2"

using namespace ff;
using namespace ff::traits;

struct MyInput {
    int a;
    int b;    
};

struct MyOutput {
    int result;
};


template <class T>
bool faas_serialize(T& b, MyInput* input) {
    b = {reinterpret_cast<char*>(input), sizeof(MyInput)};
    return false;
}

template <class T>
bool faas_deserialize(const T& b, MyOutput*& strPtr){
    strPtr = reinterpret_cast<MyOutput*>(b.first);
    return false;
}

template<typename T>
void faas_deserializealloctask(const T&, MyOutput*&) { 
}

// Bitsery serialization (obbligatoria se vuoi usare Bitsery)
template <typename S>
void serialize(S& s, MyInput& o) {
    s.value4b(o.a);
    s.value4b(o.b);
}

template <typename S>
void serialize(S& s, MyOutput& o) {
    s.value4b(o.result);
}


class Stage1 : public ff_node_t<char, MyInput> {
    public:
        MyInput* svc(char*) {
            if(NUM_TASK-- > 0) {
                std::string msg = "Sto generando il task " + std::to_string(NUM_TASK);
                PRINT_DBG(msg);              
                return new  MyInput {NUM_TASK+1, NUM_TASK};
            }
            else
                return EOS;
        }
};

class Stage3 : public ff_node_t<MyOutput,void> {
    public:

        void* svc(MyOutput* t) {
            std::string msg = "Ho raccolto il task " + std::to_string(t->result);
            PRINT_DBG(msg);                        
            delete t;
            return GO_ON;
        }

};

int main(int argc, char *argv[]) {
    
    ff_pipeline pipe;
    Stage1 s1;
    Stage3 s3;

    if(argc==2)
        NUM_TASK = atoi(argv[1]);

    if (NUM_TASK <= 0) {
        std::cerr << "Numero di task deve essere positivo.\n";
        return -1;
    }

    ff_faas_node_t<MyInput,MyOutput> faasNode(std::make_shared<std::string>(F_NAME), "/home/ferrucci/fastflow/ff/FaaS/faas_backends.json","/home/ferrucci/fastflow/ff/FaaS/faas_functions.json");
    //ff_faas_node_t<MyInput,MyOutput> faasNode2(std::make_shared<std::string>(F_NAME), "/home/ferrucci/fastflow/ff/FaaS/faas_backends.json","/home/ferrucci/fastflow/ff/FaaS/faas_functions.json");

    pipe.add_stage(&s1);
    pipe.add_stage(&faasNode);
    //pipe.add_stage(&faasNode2);
    pipe.add_stage(&s3);

    if (pipe.run_and_wait_end() < 0) {
        error("Errore durante l'esecuzione della pipeline\n");
        return -1;
    }
    
    auto stats_map = faasNode.getRealTimeStats();
    if (stats_map) {
        for (const auto& pair : *stats_map) {
            std::cout << "Key: " << pair.first << ", Value: \n" 
            << "  Function total execution time:" << pair.second->T_total << std::endl
            << "    FastFlow overhead time: " << pair.second->T_ff_overhead << std::endl
            << "    Communication time: " << pair.second->T_comm << std::endl
            << "    FAAS overhead time: " << pair.second->T_faas_overhead << std::endl   
            << "      - FAAS container initialization time:" << pair.second->T_init_container << std::endl
            << "      - FAAS function offload time:" << pair.second->T_offload << std::endl
            << "    FAAS function execution time :" << pair.second->T_fun_exec << std::endl
            << "  Warm start: " << pair.second->is_warm << std::endl 
            << "  Req msg size: " << pair.second->Msg_dim_sent << std::endl
            << "  Response msg size: " << pair.second->Msg_dim_recv << std::endl;                                            
        }
    } 

    return 0;
} 