#include <ff/ff_faas.hpp>
#include "ff/FaaS/connectors/Serverledge_connector.hpp"
#include <iostream>
#include "ff/FaaS/include/bitsery/bitsery.h"
#include <ff/FaaS/include/bitsery/traits/vector.h>
#include <cstdint>

using namespace ff;
using namespace ff::traits;
using namespace std;

class TestParams {
public:

    bool read_test(string& line) {
        stringstream ss(line);  // Stream per suddividere la riga
        string cell;

        // Lettura numero di workers in totale da generare
        getline(ss, cell, ',');
        try {
            num_workers = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore primo parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore primo parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }
        if (num_workers < 1) {
            error("Errore primo parametro: il numero deve essere strettamente positivo\n");
            return false; // Esci con codice di errore
        }

        // Numero di workers di cui fare l'offloading sul FaaS ( deve essere <= di num_workers )
        getline(ss, cell, ',');
        try {
            off_num_workers = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore secondo parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore secondo parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }
        if (off_num_workers > num_workers) {
            error("Errore secondo parametro: il numero deve essere strettamente >= 0 e <= del primo parametro\n");
            return false; // Esci con codice di errore
        }

        // Lettura numero di token da generare
        getline(ss, cell, ',');
        try {
            stream_len = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore terzo parametro : non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore terzo parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }
        if (stream_len < 1) {
            error("Errore terzo parametro: il numero deve essere strettamente positivo\n");
            return false; // Esci con codice di errore
        }

        // Tempo di interemissione dei token, in microsecondi
        getline(ss, cell, ',');
        try {
            interleave_time = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore quarto parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore quarto parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }        

        // Numero di valori da generare per token, che rappresentano i seeds
        // di una successione di valori di cui calcolare il seno
        getline(ss, cell, ',');
        try {
            num_seeds = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore quinto parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore quinto parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }
        if (num_workers < 1) {
            error("Errore quinto parametro : il numero deve essere strettamente positivo\n");
            return false; // Esci con codice di errore
        }

        // Numero di valori da generare dalla successione casuale di valori in virgola mobile 
        // di cui calcolare il seno
        getline(ss, cell, ',');
        try {
            succ_len = stoul(cell);
        }
        catch (const invalid_argument& e) {
            error("Errore sesto parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore sesto parametro: fuori dall'intervallo valido per uint32_t\n");
            return false; // Esci con codice di errore
        }

        // Nome della funzione da invocare
        getline(ss, fun_name, ',');

        // Tipo di FaaS da invocare: 
        // OpenWhisk
        // Serverledge
        getline(ss, cell, ',');
        if (cell == "OpenWhisk")
            FaasType = true;
        else {
            if (cell == "Serverledge")
                FaasType = false;
            else {
                error("Errore undicesimo parametro: può essere solo una stringa tra OpenWhisk o Serverledge\n");
                return false;
            }
        }

        // File di output del test
        getline(ss, cell, ',');
        output_file.open(cell);
        if (!output_file) {
            error("Errore dodicesimo parametro. Errore nell'apertura o creazione del logfile di output del test, al path\n");
            return false; // Esci con codice di errore
        }

        return true;
    }

    inline uint32_t getNumWorkers() const {
        return num_workers;
    }

    inline uint32_t getInterleaveTime() const {
        return interleave_time;
    }

    inline uint32_t getOffNumWorkers() const {
        return off_num_workers;
    }

    inline uint32_t getStreamLen() const {
        return stream_len;
    }

    inline uint32_t getNumSeeds() const {
        return num_seeds;
    }

    inline uint32_t getSuccLen() const {
        return succ_len;
    }

    inline bool getFaasType() const {
        return FaasType;
    }

    inline const string& getFunName() const {
        return fun_name;
    }

    inline ofstream& getOutputFile() {
        return output_file;
    }


private:
    // Numero di workers in totale da generare
    uint32_t num_workers;
    // Numero di workers di cui fare l'offloading sul FaaS ( deve essere <= di num_workers )
    uint32_t off_num_workers;
    // Numero di toker da generare
    uint32_t stream_len;
    // Tempo di interemissione dei token, in microsecondi
    uint32_t interleave_time;
    // Numero di valori da generare per token, che rappresentano i seeds
    // di una successione di valori di cui calcolare il seno
    uint32_t num_seeds;
    // Numero di valori da generare dalla successione casuale di valori in virgola mobile 
    // di cui calcolare il seno
    uint32_t succ_len;
    // Tipo di FaaS da invocare: 
    // true: OpenWhisk
    // false: Serverledge
    bool FaasType;
    // Nome della funzione da invocare
    string fun_name;
    // File di output del test
    ofstream output_file;
};

class ConfFile {

public:

    bool read_conf_CSV_file(char *filename, unique_ptr<vector<unique_ptr<TestParams>>>& tests) {
        ifstream file(filename); // Apri il file
        if (!file.is_open()) {
            string Error = "Errore nell'aprire il file di configurazione: " + string(filename);
            error(Error.c_str());
            return false;
        }
        uint32_t num_line = 0;
        string line; // Variabile per ogni riga del file
        if(!tests)
            tests = make_unique<vector<unique_ptr<TestParams>>>();

        while (getline(file, line)) { // Leggi una riga alla volta
            unique_ptr<TestParams> t = make_unique<TestParams>();
            num_line++;
            if (!(t->read_test(line)))
            {
                string Error = "Errore nel file di configurazione dei tests alla riga " + to_string(num_line) + ". Tests abortiti!\n";
                error(Error.c_str());
                file.close();
                return false;
            }
            tests->push_back(move(t));
        }

        file.close(); // Chiudi il file di configurazione dei tests
        return true;
    }
};

// Tipo del Task di input, che è un array allocato sullo heap, e la sua dimensione, quindi non contigui ( devo usare Bitsery) . Il tipo di ritorno è un semplice double. 
struct InputTask {
    vector<uint32_t> seeds_set;
    uint32_t seeds_set_size;
    uint32_t succ_len;
};

template <typename S>
void serialize(S& s, InputTask& o) {
    // Per la scrittura
    s.value4b(o.seeds_set_size);
    s.value4b(o.succ_len);
    s.container4b(o.seeds_set, o.seeds_set_size); 
}


class Emitter : public ff_node {
public:

    Emitter(TestParams* test): test(test){
        num_task = test->getStreamLen();
    }

    void* svc(void*) {
        if(test->getInterleaveTime()!=0)
            active_wait(test->getInterleaveTime());        
        if(num_task>0) {
            unique_ptr<vector<uint32_t>> generated_seeds_set = generate_seeds_set(test->getNumSeeds());
            InputTask* task = new InputTask();
            num_task--;
            task->seeds_set = std::move(*generated_seeds_set); // Spostiamo la memoria
            task->seeds_set_size = test->getNumSeeds();
            task->succ_len = test->getSuccLen();
            return task;
        }
        else
            return EOS;
    }

private:
    uint32_t num_task;
    TestParams* test;
    
    unique_ptr<vector<uint32_t>> generate_seeds_set(uint32_t num_seeds) {
            // Ottieni un seed ad alta risoluzione basato sul tempo
        uint32_t seed = static_cast<uint32_t>(chrono::high_resolution_clock::now().time_since_epoch().count());

        // Inizializzazione del generatore Mersenne Twister con il seed
        mt19937_64 rng(seed);

        // Distribuzione uniforme di valori interi tra 0 e 9999999
        uniform_int_distribution<uint32_t> dist(0, 9999999);

        unique_ptr<vector<uint32_t>> seeds_set = make_unique<vector<uint32_t>>(num_seeds);

        // Genera la sequenza casuale
        for (uint32_t i = 0; i < num_seeds; i++) {
            uint32_t unity = dist(rng) % 10;
            if (unity == 0)
                unity++;
            (*seeds_set)[i] = unity * 10000000 + dist(rng);
        }
        return seeds_set;
    }

    void active_wait(int64_t  msecs) {
        auto start = chrono::high_resolution_clock::now();
        auto end = false;
        while (!end) {
            auto elapsed = chrono::high_resolution_clock::now() - start;
            auto msec = chrono::duration_cast<chrono::microseconds>(elapsed).count();
            if (msec > msecs)
                end = true;
        }
        return;
    }
};

// generic local worker
class Worker : public ff_node_t<InputTask,float> {
public:

    float* svc(InputTask* t) {        

        chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
            
        float* mean_sin = new float(0.0f);
        for (uint32_t i = 0; i < t->seeds_set_size; ++i) 
            *mean_sin += calculate_mean_of_sine(t->succ_len, (t->seeds_set)[i]) / t->succ_len;

        chrono::duration<double> durata = chrono::high_resolution_clock::now() - start;        
        double total_exec_time = chrono::duration<double, micro>(durata).count();
        if(!stats)
            stats = make_shared<vector<double>>();

        stats->push_back(total_exec_time);
        delete t;
        return mean_sin;
    }

    shared_ptr<vector<double>> getRealTimeStats() {
        return stats;
    }

private:

    // Funzione per generare una sequenza di numeri casuali e calcolare la media del seno di ognuno dei numeri
    float calculate_mean_of_sine(uint32_t succ_len, uint32_t seed) {
        // Impostiamo il generatore con il seed
        mt19937_64 rng(seed);  // Generatore Mersenne Twister a 64 bit
        uniform_real_distribution<float> dist(0.0f, 2 * M_PI);  // Distribuzione uniforme in [0, 2π)

        float sumSine = 0.0f;

        // Generiamo la sequenza e calcoliamo la somma del seno
        for (uint32_t i = 0; i < succ_len; ++i) {
            float randomNumber = dist(rng);  // Genera un numero casuale
            sumSine += sin(randomNumber)/succ_len;  // Somma il seno del numero
        }

        // Ritorniamo la media del seno
        return sumSine ;
    }

    shared_ptr<vector<double>> stats = nullptr;
};

// the gatherer filter
class Collector : public ff_node {
public:
    void* svc(void* task) {
        double* t = (double*)task;
        delete t;
        return GO_ON;
    }
};

int main(int argc, char *argv[]) {

     if (!(argc == 3)) {
        error("Errore numero di parametri! Uso: \n test_FFConvFaas2_{NON}BLOCKING <conf_file_path>/<conf_file_name> <result_file_path>/<result_file_name>\n");
        return -1;
    }

    ConfFile c_file;
    unique_ptr<vector<unique_ptr<TestParams>>> tests;

    if (!c_file.read_conf_CSV_file(argv[1],tests)) {
        return -1;
    }
    string result_path_file = argv[2];

    ofstream result_file(argv[2]); // Apri il file dove memorizzare i risultati dei tests
    if (!result_file) {
        string err = "Errore nell'aprire o creare il file dei risultati dei tests: " + result_path_file + "\n";
        error(err.c_str());       
        return false;
    }

    bool first_test = true;
    
    for (auto& t : *tests)
    {
        if (first_test)
            first_test = false;
        else
            result_file << endl;
      
        vector<unique_ptr<ff_node>> w;

        w.push_back(make_unique<ff_faas_node_t<InputTask,float>>(std::make_shared<std::string>(t->getFunName()), "/home/ferrucci/fastflow/tests/FaaS/faas_backends.json", "/home/ferrucci/fastflow/tests/FaaS/faas_functions.json",t->getOffNumWorkers()));

        for (uint32_t i = 0; i<(t->getNumWorkers()- t->getOffNumWorkers());++i)           
            w.push_back(make_unique<Worker>());
            

        ff_Farm<InputTask> farm(move(w));

        Emitter E(t.get());
        farm.add_emitter(E);

        Collector C;
        farm.add_collector(C);

        std::chrono::high_resolution_clock::time_point T_farm_total_start = std::chrono::high_resolution_clock::now();

        if (farm.run_and_wait_end() < 0) {
            error("Errore durante l'esecuzione del Farm\n");
            return -1;
        }
        std::chrono::duration<double> T_farm_total_dur = std::chrono::high_resolution_clock::now() - T_farm_total_start;

        double T_farm_total = std::chrono::duration<double, std::micro>(T_farm_total_dur).count()/1000.0;   

        auto workers = farm.getWorkers();

        uint32_t tot_local_lines = 0, tot_FAAS_offloaded_lines = 0;
        float mean_internal_local_fun_exec_time = 0.0;
        float mean_internal_FAAS_fun_exec_time = 0.0;
        float mean_FAAS_overhead_time = 0.0;
        float mean_FF_overhead_time = 0.0;
        float mean_total_req_exec_time = 0.0;
        float mean_T_comm = 0.0;
        uint32_t mean_msg_dim = 0;
        bool first_line = true;

        for(auto& worker: workers) {
            ff_faas_node_t<InputTask,float>* faas_worker = dynamic_cast<ff_faas_node_t<InputTask,float>*> (worker);
            if(faas_worker) {
                auto stats_map = faas_worker->getRealTimeStats();
                tot_FAAS_offloaded_lines+=stats_map->size();
                for(auto& stat: *stats_map) {
                    if (first_line)
                        first_line = false;
                    else
                        (t->getOutputFile()) << endl;
                    // 1 if the call is to the FAAS 
                    // + sent msg size ( in bytes )
                    // + warm or not 
                    // + T_comm in milliseconds 
                    // + FAAS_overhead in milliseconds
                    // + FF_overhead in milliseconds
                    // + int_fun_exec_time in milliseconds 
                    // + total_execution_time in milliseconds
                        
                    (t->getOutputFile()) << "1," << stat.second->Msg_dim_sent << ","
                    << (stat.second->is_warm ? "1,":"0,")
                    << stat.second->T_comm/1000.0 << ","
                    << stat.second->T_faas_overhead/1000.0 << ","
                    << stat.second->T_req_ff_overhead/1000.0 << ","
                    << stat.second->T_fun_exec/1000.0 << ","
                    << stat.second->T_req_total/1000.0; 
                    
                    mean_internal_FAAS_fun_exec_time += stat.second->T_fun_exec/1000.0;   
                    mean_FAAS_overhead_time += stat.second->T_faas_overhead/1000.0;
                    mean_FF_overhead_time += stat.second->T_req_ff_overhead/1000.0;
                    mean_total_req_exec_time += stat.second->T_req_total/1000.0;
                    mean_msg_dim += stat.second->Msg_dim_sent;
                    mean_T_comm += stat.second->T_comm/1000.0;
                }
            }
            else {
                Worker* local_worker = dynamic_cast<Worker*>(worker);
                auto stats_vector = local_worker->getRealTimeStats();
                tot_local_lines+=stats_vector->size();
                for(double& stat: *stats_vector) {
                    if (first_line)
                        first_line = false;
                    else
                        (t->getOutputFile()) << endl;
                    (t->getOutputFile()) << "0,0,0,0,0,0," << stat / 1000.0 << ",0";
                    mean_internal_local_fun_exec_time += stat/1000.0;   
                }
            }
        }

        // Calculation of the means of results
        if (tot_local_lines!= 0)
            mean_internal_local_fun_exec_time /= tot_local_lines;

        if (tot_FAAS_offloaded_lines != 0) {
            mean_msg_dim /= tot_FAAS_offloaded_lines;
            mean_internal_FAAS_fun_exec_time /= tot_FAAS_offloaded_lines;
            mean_FAAS_overhead_time /= tot_FAAS_offloaded_lines;
            mean_FF_overhead_time /= tot_FAAS_offloaded_lines;
            mean_total_req_exec_time /= tot_FAAS_offloaded_lines;
        }

        result_file << t->getNumWorkers() << "," << t->getOffNumWorkers() << "," << t->getStreamLen() << "," << mean_msg_dim << "," << mean_internal_local_fun_exec_time << "," << mean_internal_FAAS_fun_exec_time << "," << mean_T_comm << "," << mean_FAAS_overhead_time << "," << mean_FF_overhead_time << "," << mean_total_req_exec_time << "," << farm.ffTime();

        cout << "Test with " << t->getNumWorkers() << " workers, of which " << t->getOffNumWorkers() << " offloaded to the FaaS, and " << t->getStreamLen() << " tasks: DONE" << endl;

        t->getOutputFile().close();

        this_thread::sleep_for(chrono::seconds(2));
    }
    result_file.close();
    return 0;
} 