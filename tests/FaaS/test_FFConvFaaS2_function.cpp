#include <ff/FaaS/adapters/Serverledge_adapter.hpp>
#include <ff/FaaS/ff_faas_function.hpp>
#include <iostream>
#include "ff/FaaS/include/bitsery/bitsery.h"
#include <ff/FaaS/include/bitsery/traits/vector.h>
#include <cstdint>

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

class MyFaaSFunction : public ff::ff_faas_function<InputTask, float, Serverledge_adapter> {
public:
    float* svc(InputTask* t) override {        
        float* mean_sin = new float(0.0f);
        for (uint32_t i = 0; i < t->seeds_set_size; ++i) 
            *mean_sin += calculate_mean_of_sine(t->succ_len, (t->seeds_set)[i]) / t->succ_len;

        delete t;
        return mean_sin;        
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
};

int main(void) {
    MyFaaSFunction faasFunction;
    return faasFunction.run_and_wait_end();
}