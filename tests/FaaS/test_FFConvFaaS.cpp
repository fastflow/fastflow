 // farm with emitter and collector explicitly defined as multi-output and
 // multi-input, respectively
 /*
  *
  *               |--> Worker -->|
  *               |              |
  *   Emitter --> |--> Worker -->|--> Collector
  *               |              |
  *               |--> Worker -->|
  *
  * Emitter can be either a standard ff_node or a ff_monode
  * Collector can be either a standard ff_node or a ff_minode
  *
  * The same topology can be constructed by using 2 building blocks:
  *  pipe(Emitter, A2A(Worker, Collector))
  *
  */
  /* Author: Luca Ferrucci
   *
   */

#include <vector>
#include <iostream>
#include <fstream>
#include <ff/ff.hpp>
#include <ff/make_unique.hpp>
#include <string>
#include <curl/curl.h>
#include <chrono>
#include <sstream>
#include <regex>
#include <cmath>
#include <random>
#include <iomanip>
#include <thread>
#include "rapidjson/document.h"
#include "rapidjson/writer.h" 
#include <rapidjson/error/en.h>
using namespace ff;

#define DEBUG 1
#define NUM_REP_COMM_TEST 20
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

using namespace ff;
using namespace std;
using namespace rapidjson;


// Inizializzazione del mutex per stampare su cout in thread_safe

static pthread_mutex_t cout_mutex  PTHREAD_MUTEX_INITIALIZER;

static void print(const ostringstream& message) {
    pthread_mutex_lock(&cout_mutex);
    cout << message.str() << endl;
    pthread_mutex_unlock(&cout_mutex);
}

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
            error("Errore primo parametro: fuori dall'intervallo valido per unsigned long\n");
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
            error("Errore secondo parametro: fuori dall'intervallo valido per unsigned long\n");
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
            error("Errore terzo parametro: fuori dall'intervallo valido per unsigned long\n");
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
            error("Errore quarto parametro: fuori dall'intervallo valido per unsigned long\n");
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
            error("Errore quinto parametro: fuori dall'intervallo valido per unsigned long\n");
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
            error("Errore sesto parametro: fuori dall'intervallo valido per unsigned long\n");
            return false; // Esci con codice di errore
        }

        // Indirizzo ip dove si trova in ascolto il servizio WebAPI del server FaaS ( stesso ip del server HTTP per il calcolo dei tempi di comunicazione )
        getline(ss, http_ip, ',');
        regex pattern("^(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\\.(25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$");
        if (!regex_match(http_ip, pattern)) {
            error("Errore settimo parametro: la stringa non è un indirizzo ipv4 valido\n");
            return false; // Esci con codice di errore
        }

        //TODO: da controllare
        string port;
        int check_port;
        getline(ss, port, ',');
        try {
            check_port = stoi(port);
        }
        catch (const invalid_argument& e) {
            error("Errore ottavo parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore ottavo parametro: fuori dall'intervallo valido per int\n");
            return false; // Esci con codice di errore
        }
        if ((check_port < 1) || (check_port > 65535)) {
            error("Errore ottavo parametro: il numero deve essere tra 1 e 65535\n");
            return false; // Esci con codice di errore
        }

        getline(ss, http_port, ',');
        try {
            check_port = stoi(http_port);
        }
        catch (const invalid_argument& e) {
            error("Errore nono parametro: non è un numero valido\n");
            return false; // Esci con codice di errore
        }
        catch (const out_of_range& e) {
            error("Errore nono parametro: fuori dall'intervallo valido per int\n");
            return false; // Esci con codice di errore
        }
        if ((check_port < 1) || (check_port > 65535)) {
            error("Errore nono parametro: il numero deve essere tra 1 e 65535\n");
            return false; // Esci con codice di errore
        }

        // Nome della funzione da invocare
        string fun_name;
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

        // Calcolo dell'URL da invocare sulla base del tipo di server FaaS
        if (FaasType)
            URL = "https://" + http_ip + ":" + port + "/api/v1/namespaces/_/actions/" + fun_name + "?blocking=true";
        else
            URL = "http://" + http_ip + ":" + port + "/invoke/" + fun_name;

        // Token di autenticazione ( solo per OpenWhisk )
        getline(ss, auth_token, ',');

        // File di output del test
        getline(ss, cell, ',');
        output_file.open(cell);
        if (!output_file) {
            error("Errore dodicesimo parametro. Errore nell'apertura o creazione del logfile di output del test, al path\n");
            return false; // Esci con codice di errore
        }

        return true;
    }

    inline unsigned long int getNumWorkers() const {
        return num_workers;
    }

    inline unsigned long int getInterleaveTime() const {
        return interleave_time;
    }

    inline unsigned long int getOffNumWorkers() const {
        return off_num_workers;
    }

    inline unsigned long int getStreamLen() const {
        return stream_len;
    }

    inline unsigned long int getNumSeeds() const {
        return num_seeds;
    }

    inline const string& getHttpIp() const {
        return http_ip;
    }

    inline const string& getHttpPort() const {
        return http_port;
    }

    inline unsigned long int getSuccLen() const {
        return succ_len;
    }

    inline bool getFaasType() const {
        return FaasType;
    }

    inline const string& getURL() const {
        return URL;
    }

    inline const string& getAuthToken() const {
        return auth_token;
    }

    inline ofstream& getOutputFile() {
        return output_file;
    }

    inline float getRTT() const {
        return RTT;
    }

    inline void setRTT(float RTT)  {
        this->RTT = RTT;
    }

private:
    // Numero di workers in totale da generare
    unsigned long int num_workers;
    // Numero di workers di cui fare l'offloading sul FaaS ( deve essere <= di num_workers )
    unsigned long int off_num_workers;
    // Numero di toker da generare
    unsigned long int stream_len;
    // Tempo di interemissione dei token, in microsecondi
    unsigned long int interleave_time;
    // Numero di valori da generare per token, che rappresentano i seeds
    // di una successione di valori di cui calcolare il seno
    unsigned long int num_seeds;
    // Numero di valori da generare dalla successione casuale di valori in virgola mobile 
    // di cui calcolare il seno
    unsigned long int succ_len;
    // URL da invocare: dipende dal tipo di FaaS
    string URL;
    // Ip del server HTTP per calcolare il tempo di comunicazione ( stesso ip del server FAAS )
    string http_ip;
    // Porta del server HTTP per calcolare il tempo di comunicazione
    string http_port;
    // Tipo di FaaS da invocare: 
    // true: OpenWhisk
    // false: Serverledge
    bool FaasType;
    // Token di autenticazione ( solo per OpenWhisk )
    string auth_token;
    // File di output del test
    ofstream output_file;
    // Round trip time ( tempo di comunicazione ), calcolato all'inizio,  con il FAAS
    float RTT;
};

class Task {
public:Task(long int index, TestParams* test_params) : test_params(test_params), index(index) {

    // Calcolo array con i seed 
    seeds_set = std::make_unique< unsigned long[]>(test_params->getNumSeeds());

    // Ottieni un seed ad alta risoluzione basato sul tempo
    unsigned long seed = static_cast<unsigned long>(chrono::high_resolution_clock::now().time_since_epoch().count());

    // Inizializzazione del generatore Mersenne Twister con il seed
    mt19937_64 rng(seed);

    // Distribuzione uniforme di valori interi tra 0 e 9999999
    uniform_int_distribution<unsigned long> dist(0, 9999999);

    // Genera la sequenza casuale
    for (unsigned long i = 0; i < test_params->getNumSeeds(); i++) {
        unsigned long unity = dist(rng) % 10;
        if (unity == 0)
            unity++;
        seeds_set[i] = unity * 10000000 + dist(rng);
    }
}

    inline void setLogline(const string& logline) {
        this->logline = logline;
    };

    inline int getIndex() const {
        return index;
    }

    inline string getLogline() {
        return logline;
    }

    inline TestParams* getTestParams() {
        return test_params;
    }

    inline const unsigned long* getSeedsSet() const {
        return seeds_set.get();
    }

    inline void setMeanSine(float mean_sine) {
        this->mean_sine = mean_sine;
    }

    inline float getMeanSine() {
        return mean_sine;
    }

private:
    TestParams* test_params;
    long int index;
    string logline;
    // Array che contiene i seed generati casualmente
    std::unique_ptr< unsigned long[]> seeds_set;
    float mean_sine;
};

// generic worker
class Worker : public ff_node_t<Task> {
public:

    ~Worker() {
        // Clean up and release resources
        curl_easy_cleanup(curl);
        curl_slist_free_all(headers);
    }

    Task* svc(Task* t) {        
        TestParams* test_params = t->getTestParams();

        // Se vero, il task sta su un worker locale, altrimenti va mandato al FaaS
        if (get_my_id() >= test_params->getOffNumWorkers()) {
            chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();
            
            float mean_sin = 0.0f;
            for (unsigned long i = 0; i < test_params->getNumSeeds(); ++i) 
                mean_sin += calculate_mean_of_sine(test_params->getSuccLen(), (t->getSeedsSet())[i]) / test_params->getNumSeeds();

            // Scrivi nel Task il risultato finale della media delle medie dei seni delle sequenze casuali
            // di numeri in floating point
            t->setMeanSine(mean_sin);

            chrono::duration<double> durata = chrono::high_resolution_clock::now() - start;
            auto total_exec_time = chrono::duration<double, micro>(durata).count();
            t->setLogline(to_string(get_my_id()) + "," + to_string(t->getIndex()) + ",0,0," + to_string(t->getMeanSine()) + ",0,0,0,0,0," + to_string(total_exec_time / 1000.0) + ",0");
            #if DEBUG == 1
            ostringstream msg;
            msg << "Worker number " << get_my_id() << ". Task number " << t->getIndex() << ". Esecuzione locale con tempo totale:" << total_exec_time / 1000.0 << endl;
            print(msg);
            #endif
            return t;
        }
        else {
            call_FaaS(*t,*test_params);
            return t;
        }
    }

    static string serialize_params(const unsigned long* seeds_set, unsigned long seeds_size, unsigned long seq_length) {
        // Crea un documento RapidJSON e impostalo come oggetto
        Document doc;
        doc.SetObject();
        Document::AllocatorType& allocator = doc.GetAllocator();

        // Crea un array JSON per "seeds"
        Value seeds(rapidjson::kArrayType);
        for (unsigned long i = 0; i < seeds_size; ++i) {
            // Aggiungi ogni seme all'array (i numeri interi vengono aggiunti direttamente)
            seeds.PushBack(static_cast<uint64_t>(seeds_set[i]), allocator);
        }
        // Aggiungi l'array "seeds" al documento
        doc.AddMember("seeds", seeds, allocator);

        // Aggiungi il membro "seq_length"
        doc.AddMember("seq_length", static_cast<uint64_t>(seq_length), allocator);

        // Converte il documento in una stringa JSON
        StringBuffer buffer;
        Writer<StringBuffer> writer(buffer);
        doc.Accept(writer);

        return buffer.GetString();
    }

private:

    // Funzione per generare una sequenza di numeri casuali e calcolare la media del seno di ognuno dei numeri
    float calculate_mean_of_sine(unsigned long succ_len, unsigned long seed) {
        // Impostiamo il generatore con il seed
        mt19937_64 rng(seed);  // Generatore Mersenne Twister a 64 bit
        uniform_real_distribution<float> dist(0.0f, 2 * M_PI);  // Distribuzione uniforme in [0, 2π)

        float sumSine = 0.0f;

        // Generiamo la sequenza e calcoliamo la somma del seno
        for (unsigned long i = 0; i < succ_len; ++i) {
            float randomNumber = dist(rng);  // Genera un numero casuale
            sumSine += sin(randomNumber)/succ_len;  // Somma il seno del numero
        }

        // Ritorniamo la media del seno
        return sumSine ;
    }

    bool init_FaaS_comm(Task& t, TestParams& test_params) {
        // Initialize the libcurl library
        if (!curl) {

            curl = curl_easy_init();
            if (!curl) {
                t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                #if DEBUG == 1
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Fallimento nell\'inizializzare curl!" << endl;
                print(msg);
                #endif
                return false;
            }

            // Set the API endpoint and method
            curl_easy_setopt(curl, CURLOPT_URL, (test_params.getURL()).c_str());
            curl_easy_setopt(curl, CURLOPT_POST, 1L);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
            curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);

            //true: OpenWhisk
            if (test_params.getFaasType())
            {
                // Set the authentication header to an empty string
                const string& auth_key = test_params.getAuthToken();
                headers = curl_slist_append(headers, ("Authorization: Basic " + auth_key).c_str());
                curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
            }

            headers = curl_slist_append(headers, "Content-Type: application/json");
            // Set the write callback to capture the API response
            curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, write_callback);
            curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

            curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);  // Necessario per abilitare il debug
            curl_easy_setopt(curl, CURLOPT_DEBUGFUNCTION, msg_size_callback_static);  // Registra la callback
            curl_easy_setopt(curl, CURLOPT_DEBUGDATA, this);
            return true;
        }
        return true;
    }

    void call_FaaS(Task& t, TestParams& test_params) {

        try {
            chrono::high_resolution_clock::time_point start_tot_exec_time = chrono::high_resolution_clock::now();

            if (!init_FaaS_comm(t, test_params)) {
                return;
            }

            string json_payload;
            header_size = 0;

            //true: OpenWhisk
            if (test_params.getFaasType()) {
                json_payload = serialize_params(t.getSeedsSet(), test_params.getNumSeeds(), test_params.getSuccLen());
                #if DEBUG == 1
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Inizio richiesta a Openwhisk. Parametri: " << json_payload << endl;
                print(msg);
                #endif
            }
            //false: Serverledge
            else
            {
                json_payload = "{\"Params\" : " + serialize_params(t.getSeedsSet(), test_params.getNumSeeds(), test_params.getSuccLen()) + "}";
                #if DEBUG == 1
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Inizio richiesta a Serverledge. Parametri: " << json_payload << endl;
                print(msg);
                #endif
            }

            // Set the JSON payload
            curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());
            curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.size());
            auto params_preparation_time = chrono::duration<double, micro>(chrono::high_resolution_clock::now() - start_tot_exec_time).count();

            // Perform the API request
            res = curl_easy_perform(curl);

            chrono::high_resolution_clock::time_point start_response_time = chrono::high_resolution_clock::now();            

            #if DEBUG == 1
            ostringstream msg;
            msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Dimensione payload della richiesta CURL: " << json_payload.size() << ", dimensione dell'header della richiesta CURL: " << header_size << " per un totale di " << json_payload.size()+ header_size << " bytes" << endl;
            print(msg);
            #endif  

            //TODO: da cambiare!
            // Check for errors
            if (res != CURLE_OK) {
                string curl_error = curl_easy_strerror(res);
                t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                #if DEBUG == 1
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore di invocazione API di curl: " << curl_error << endl;
                print(msg);
                #endif
                return;
            }

            Document result;
            rapidjson::ParseResult parseResult = result.Parse(data.c_str());
            if (result.HasParseError()) {
                t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                #if DEBUG == 1
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore parsing dei parametri di ingresso Json: " << GetParseError_En(parseResult.Code()) << u8" alla riga " << parseResult.Offset() << endl;
                print(msg);
                #endif
                return;
            }

            float initTime = 0.0;
            float duration = 0;
            string error = "";
            float functionExecTime = 0.0;
            float meanSin = 0.0;
            bool success = true;

            // Openwhisk
            if (test_params.getFaasType()) {

                if (result["response"]["result"].HasMember("Error")) {
                    #if DEBUG == 1
                    ostringstream msg;
                    msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore di esecuzione della funzione in OpenWhisk: " << result["response"]["result"]["Error"].GetString() << endl;
                    print(msg);
                    #endif
                    t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                    return;
                }
                meanSin = result["response"]["result"]["MeanSin"].GetFloat();

                duration = result["duration"].GetInt64();
                functionExecTime = result["response"]["result"]["FunctionExecutionTime"].GetFloat(); 
                success = result["response"]["success"].GetBool();
                // Estrai 'initTime' da 'annotations'
                for (const auto& annotation : result["annotations"].GetArray()) {
                    if (annotation["key"] == "initTime") {
                        initTime = annotation["value"].GetInt();
                        break;
                    }
                }
                if (!success) {
                    #if DEBUG == 1
                    ostringstream msg;
                    msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore di esecuzione della funzione in OpenWhisk: esecuzione della funzione senza successo!" << endl;
                    print(msg);
                    #endif
                    t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                    return;
                }
            }
            // Serverledge
            else {
                success = result["Success"].GetBool();
                if (!success) {
                    #if DEBUG == 1
                    ostringstream msg;
                    msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore di esecuzione della funzione in Serverledge: esecuzione della funzione senza successo!" << endl;
                    print(msg);
                    #endif
                    t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                    return;
                }

                duration = result["ResponseTime"].GetFloat() * 1000.0;
                if (!result["IsWarmStart"].GetBool())
                    initTime = result["InitTime"].GetFloat() * 1000.0;

                string resultStr = result["Result"].GetString();

                Document fun_result;
                rapidjson::ParseResult fun_parseResult = fun_result.Parse(resultStr.c_str());

                if (fun_result.HasParseError()) {
                    #if DEBUG == 1
                    ostringstream msg;
                    msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore parsing dei parametri di ingresso Json a Serverledge: " << GetParseError_En(fun_parseResult.Code()) << u8" alla riga " << fun_parseResult.Offset() << endl;
                    print(msg);
                    #endif
                    t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                    return;
                }

                if (fun_result.HasMember("Error")) {
                    #if DEBUG == 1
                    ostringstream msg;
                    msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Errore di esecuzione della funzione in Serverledge: " << result["response"]["result"]["Error"].GetString() << endl;
                    print(msg);
                    #endif
                    t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
                    return;
                }

                meanSin = fun_result["MeanSin"].GetFloat();
                functionExecTime = fun_result["FunctionExecutionTime"].GetFloat();
            }

            chrono::high_resolution_clock::time_point end_response_time = chrono::high_resolution_clock::now();

            auto total_response_time = chrono::duration<double, micro>(end_response_time - start_response_time).count();

            auto total_exec_time = chrono::duration<double, micro>(end_response_time - start_tot_exec_time).count();

            float FAAS_overhead = duration - functionExecTime + initTime;

            float FF_overhead = (total_response_time + params_preparation_time) / 1000.0;

            if (initTime != 0)
                t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,0," + to_string(meanSin) + "," + to_string(json_payload.size() + header_size) + ",1," + to_string(test_params.getRTT()) + "," + to_string(FAAS_overhead) + "," + to_string(FF_overhead) + "," + to_string(functionExecTime) + "," + to_string(total_exec_time / 1000.0));
            else
                t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,0," + to_string(meanSin) + "," + to_string(json_payload.size() + header_size) + ",0," + to_string(test_params.getRTT()) + "," + to_string(FAAS_overhead) + "," + to_string(FF_overhead) + "," + to_string(functionExecTime) + "," + to_string(total_exec_time / 1000.0));

            #if DEBUG == 1
            if (test_params.getFaasType()) {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Openwhisk " << endl;
                print(msg);
            }
            else {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Serverledge " << endl;
                print(msg); 
            }
            msg.str("");
            msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Risultato chiamata: \n" << data <<  " con tempo totale: " << total_exec_time / 1000.0  << endl;
            print(msg);
            #endif
            return;
        }
        catch (const exception& e) {  // Intercetta solo le eccezioni standard
            #if DEBUG == 1
            if (test_params.getFaasType()) {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Openwhisk con fallimento! Eccezione: " << e.what() << endl;
                print(msg);
            }
            else {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Serverledge con fallimento! Eccezione: " << e.what() << endl;
                print(msg);
            }
            #endif
            t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
            return;
        }
        catch (...) {  // Intercetta tutte le altre eccezioni

            #if DEBUG == 1
            if (test_params.getFaasType()) {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Openwhisk con fallimento! Eccezione generica!" << endl;
                print(msg);
            }
            else {
                ostringstream msg;
                msg << "Worker number " << get_my_id() << ". Task number " << t.getIndex() << ". Finita richiesta a Serverledge con fallimento! Eccezione generica!" << endl;
                print(msg);
            }
            #endif
            t.setLogline(to_string(get_my_id()) + "," + to_string(t.getIndex()) + ",1,1,0.0,0,0,0,0,0,0,0");
            return;
        }
    }


    // Funzione privata per intercettare gli header
    int msg_size_callback(CURL* /*handle*/, curl_infotype type, char* /*data*/, size_t size, void* /*userp*/) {
        if (type == CURLINFO_HEADER_OUT) {
            header_size = size; // Salva la dimensione dell'header
        }
        return 0; // Indica che la callback è stata completata senza errori
    }

    static size_t write_callback(void* ptr, size_t size, size_t nmemb, string& data) {
        size_t total_size = size * nmemb;
        // Copiare i dati in 'data' senza appenderli
        data.assign((char*)ptr, total_size);
        return total_size;       
    }

    // Funzione statica di aiuto per la callback
    static int msg_size_callback_static(CURL* handle, curl_infotype type, char* data, size_t size, void* userp) {
        // Cast del userp in Worker*
        Worker* worker = static_cast<Worker*>(userp);
        if (worker) {
            return worker->msg_size_callback(handle, type, data, size, userp); // Chiamata alla funzione membro
        }
        return 0;
    }
     
    size_t header_size = 0;
    CURL* curl = nullptr;
    CURLcode res;
    string data; // Variable to store the API response
    struct curl_slist* headers = nullptr;
};

// the gatherer filter
class Collector : public ff_node {
public:
    Collector(unsigned long int stream_len): stream_len(stream_len) {
        loglines = new string [stream_len];
    }

    ~Collector() {
        delete[] loglines;
    }

    void* svc(void* task) {
        Task* t = (Task*)task;
        loglines[t->getIndex()] = t->getLogline();
        delete t;
        return GO_ON;
    }

    inline string* getLogLines() {
        return loglines;
    }

private:
    //ofstream& logfile;
    string* loglines = nullptr;
    unsigned long int stream_len;
};

// the load-balancer filter
class Emitter : public ff_node {
public:

    Emitter(TestParams* test): test(test){
        num_task = test->getStreamLen();
    }

    void* svc(void*) {
        if(test->getInterleaveTime()!=0)
            active_wait(test->getInterleaveTime());        
        if(num_task>0)
            return new Task(--num_task, test);
        else
            return EOS;
    }

private:
    unsigned long num_task;
    TestParams* test;
    
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

class ConfFile {

public:
    bool read_conf_CSV_file(char *filename) {
        ifstream file(filename); // Apri il file
        if (!file.is_open()) {
            string Error = "Errore nell'aprire il file di configurazione: " + string(filename);
            error(Error.c_str());
            return false;
        }
        unsigned long int num_line = 0;
        string line; // Variabile per ogni riga del file
        while (getline(file, line)) { // Leggi una riga alla volta
            TestParams* t = new TestParams();
            num_line++;
            if (!(t->read_test(line)))
            {
                string Error = "Errore nel file di configurazione dei tests alla riga " + to_string(num_line) + ". Tests abortiti!\n";
                error(Error.c_str());
                delete t;
                file.close();
                return false;
            }
            tests.push_back(t);
        }

        file.close(); // Chiudi il file di configurazione dei tests
        return true;
    }

    inline vector<TestParams*>& getTests() {
        return tests;
    }

    ~ConfFile() {
        for (TestParams* t : tests)
            delete t;  // Dealloca la memoria per ogni oggetto
    }

private:
    vector <TestParams*> tests;
};


static size_t communication_test_callback(void* ptr, size_t size, size_t nmemb, string& data) {
    size_t total_size = size * nmemb;
    // Copiare i dati in 'data' senza appenderli
    data.assign((char*)ptr, total_size);

    return total_size;
}

#if DEBUG == 1
static size_t header_size = 0;

// Callback per intercettare gli header inviati da cURL
static int communication_test_size_callback(CURL* , curl_infotype type, char* , size_t size, void* ) {
    if (type == CURLINFO_HEADER_OUT) {  // Header della richiesta
        header_size = size;
    }
    return 0;
}
#endif

static float calculate_RTT(TestParams& t) {
    // Calcolo il tempo medio di comunicazione, su un certo numero di invocazioni al server HTTP che si trova sullo stesso nodo del server FAAS
    if (t.getOffNumWorkers() > 0) {
        #if DEBUG == 1
        ostringstream msg;
        #endif  
        float t_rtt = 0.0f;
        string http_url = "http://" + t.getHttpIp() + ":" + t.getHttpPort();
        CURL* curl = nullptr;
        CURLcode res;
        string data; // Variable to store the API response

        // Calcolo array con i seed 
        unsigned long* seeds_set = new unsigned long[t.getNumSeeds()];

        // Ottieni un seed ad alta risoluzione basato sul tempo
        unsigned long seed = static_cast<unsigned long>(chrono::high_resolution_clock::now().time_since_epoch().count());

        // Inizializzazione del generatore Mersenne Twister con il seed
        mt19937_64 rng(seed);

        // Distribuzione uniforme di valori interi tra 0 e 9999999
        uniform_int_distribution<unsigned long> dist(0, 9999999);

        // Genera la sequenza casuale
        for (unsigned long i = 0; i < t.getNumSeeds(); i++) {
            unsigned long unity = dist(rng) % 10;
            if (unity == 0)
                unity++;
            seeds_set[i] = unity * 10000000 + dist(rng);
        }
        string json_payload;
        if(t.getFaasType() == 1)
            json_payload = Worker::serialize_params(seeds_set, t.getNumSeeds(), t.getSuccLen());
        else
            json_payload = u8"{\"Params\" : " + Worker::serialize_params(seeds_set, t.getNumSeeds(), t.getSuccLen()) + "}";

        #if DEBUG == 1
        msg.str("");
        msg << " Payload del msg spedito: " << json_payload << endl;
        print(msg);
        #endif 


        curl = curl_easy_init();
        if (!curl) {
            error("Fallimento nell\'inizializzare curl: comunicazione col server HTTP di calcolo dei tempi di comunicazione fallita!\n");
            delete[] seeds_set;
            return -1;
        }

        curl_easy_setopt(curl, CURLOPT_URL, (http_url.c_str()));
        curl_easy_setopt(curl, CURLOPT_POST, 1L);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYPEER, false);
        curl_easy_setopt(curl, CURLOPT_SSL_VERIFYHOST, 0);

        // Set the JSON payload (if required)
        curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_payload.c_str());

        // Set the write callback to capture the API response
        curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, communication_test_callback);
        curl_easy_setopt(curl, CURLOPT_WRITEDATA, &data);

        #if DEBUG == 1
        curl_easy_setopt(curl, CURLOPT_VERBOSE, 1L);  // Necessario per abilitare il debug
        curl_easy_setopt(curl, CURLOPT_POSTFIELDSIZE, json_payload.size());
        curl_easy_setopt(curl, CURLOPT_DEBUGFUNCTION, communication_test_size_callback);  // Registra la callback
        #endif   

        // Perform the API request
        if(NUM_REP_COMM_TEST>0)
            res = curl_easy_perform(curl);

        // Check for errors 
        if (res != CURLE_OK) {
            string curl_error = curl_easy_strerror(res);
            string err = "Errore di invocazione API di curl nel calcolo del RTT : " + curl_error + "\n";
            error(err.c_str());
            // Clean up and release resources
            curl_easy_cleanup(curl);
            delete[] seeds_set;
            return -1;
        }

        for (int i = 1; i < NUM_REP_COMM_TEST; i++) {
           
            // Set the API endpoint and method

            chrono::high_resolution_clock::time_point start = chrono::high_resolution_clock::now();

            // Perform the API request
            res = curl_easy_perform(curl);

            chrono::duration<double> durata = chrono::high_resolution_clock::now() - start;
            auto durata_musec = chrono::duration_cast<chrono::microseconds>(durata).count();

            #if DEBUG == 1        
            msg.str("");
            msg << " Dimensione payload del msg spedito: " << json_payload.size() <<  ", dimensione header del msg spedito: " << header_size << " per un totale di " << json_payload.size() + header_size << " bytes" << endl;
            print(msg);
            #endif   

            // Check for errors 
            if (res != CURLE_OK) {
                string curl_error = curl_easy_strerror(res);
                string err = "Errore di invocazione API di curl nel calcolo del RTT : " + curl_error + "\n";
                error(err.c_str());
                // Clean up and release resources
                curl_easy_cleanup(curl);
                delete[] seeds_set;
                return -1;
            }

            if (data.compare("") == 1) {
                #if DEBUG == 1
                msg.str("");
                msg << "RTT: " << (durata_musec / 1000.0f) - stof(data) << endl;
                print(msg);
                #endif
                t_rtt += ((durata_musec / 1000.0f) - stof(data)) / NUM_REP_COMM_TEST;
            }
        }

        // Clean up and release resources
        curl_easy_cleanup(curl);


        #if DEBUG == 1
        msg.str("");
        msg << " Media del RTT: " << t_rtt << endl;
        print(msg);
        #endif

        delete[] seeds_set;
        return t_rtt;
    }    
    return 0;
}

int main(int argc, char* argv[]) {
  
    if (!(argc == 3)) {
        error("Errore numero di parametri! Uso: \n test_FFConvFaas_{NON}BLOCKING <conf_file_path>/<conf_file_name> <result_file_path>/<result_file_name>\n");
        return -1;
    }

    ConfFile c_file;

    if (!c_file.read_conf_CSV_file(argv[1])) {
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
    curl_global_init(CURL_GLOBAL_DEFAULT);

    for (TestParams* t : c_file.getTests())
    {
        float RTT = calculate_RTT(*t);
        if (RTT == -1) {
            curl_global_cleanup();
            return -1;
        }

        t->setRTT(RTT);    

        if (first_test)
            first_test = false;
        else
            result_file << endl;
      
        vector<unique_ptr<ff_node>> w;
        for (unsigned long int i = 0;i < t->getNumWorkers();++i)
            w.push_back(std::make_unique<Worker>());

        ff_Farm<Task> farm(std::move(w));

        Emitter E(t);
        farm.add_emitter(E);

        Collector C(t->getStreamLen());
        farm.add_collector(C);

        if (farm.run_and_wait_end() < 0) {
            error("Errore durante l'esecuzione del Farm\n");
            curl_global_cleanup();
            return -1;
        }

        #if DEBUG == 1    
        ostringstream msg;
        msg << "DONE, time= " << farm.ffTime() << " (ms)\n" << endl;
        print(msg);
        #endif

        string* loglines = C.getLogLines();
        unsigned long tot_local_lines = 0, tot_FAAS_offloaded_lines = 0;
        float mean_internal_local_fun_exec_time = 0.0;
        float mean_internal_FAAS_fun_exec_time = 0.0;
        float mean_FAAS_overhead_time = 0.0;
        float mean_FFprep_overhead_time = 0.0;
        float mean_total_req_exec_time = 0.0;
        unsigned long mean_msg_dim = 0;
        bool first_line = true;

        for (unsigned long int i = 0;i < t->getStreamLen(); i++) {
            if (first_line)
                first_line = false;
            else
                (t->getOutputFile()) << endl;


            // Scrivi linea sul file di output del test
            (t->getOutputFile()) << loglines[i];

            // Crea un array di 12 stringhe per memorizzare i token
            string tokens[12];

            // Usa un stringstream per scorrere la stringa
            stringstream ss(loglines[i]);
            string token;

            int j = 0;
            // Estrai i token separati dalla virgola
            while (getline(ss, token, ',') && j < 12) {
                tokens[j] = token;  // Memorizza ogni token nell'array
                j++;
            }

            // Ignoro la linea se c'è stato un errore: 1 errore, 0 altrimenti. Conto solo le linee senza errori
            if (tokens[3].compare("0") == 0) {
                // Controllo se la linea fa riferimento ad una esecuzione locale o al FAAS : 0 locale, 1 FAAS
                // FAAS
                if (tokens[2].compare("1") == 0) {
                    tot_FAAS_offloaded_lines++;
                    mean_internal_FAAS_fun_exec_time += stof(tokens[10]);
                    mean_FAAS_overhead_time += stof(tokens[8]);
                    mean_FFprep_overhead_time += stof(tokens[9]);
                    mean_total_req_exec_time += stof(tokens[11]);
                    mean_msg_dim += stoul(tokens[5]);
                }
                // Locale
                else {
                    tot_local_lines++;
                    mean_internal_local_fun_exec_time += stof(tokens[10]);
                }
            }

        }

        // Calcolo le medie dei risultati
        if (tot_local_lines != 0)
            mean_internal_local_fun_exec_time /= tot_local_lines;

        if (tot_FAAS_offloaded_lines != 0) {
            mean_msg_dim /= tot_FAAS_offloaded_lines;
            mean_internal_FAAS_fun_exec_time /= tot_FAAS_offloaded_lines;
            mean_FAAS_overhead_time /= tot_FAAS_offloaded_lines;
            mean_FFprep_overhead_time /= tot_FAAS_offloaded_lines;
            mean_total_req_exec_time /= tot_FAAS_offloaded_lines;
        }
        result_file << t->getNumWorkers() << "," << t->getOffNumWorkers() << "," << t->getStreamLen() << "," << mean_msg_dim << "," << mean_internal_local_fun_exec_time << "," << mean_internal_FAAS_fun_exec_time << "," << RTT << "," << mean_FAAS_overhead_time << "," << mean_FFprep_overhead_time << "," << mean_total_req_exec_time << "," << farm.ffTime();

        t->getOutputFile().close();

        this_thread::sleep_for(chrono::seconds(2));
    }
    result_file.close();
    curl_global_cleanup();
    return 0;
}