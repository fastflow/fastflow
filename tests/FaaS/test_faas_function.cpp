#include <ff/FaaS/adapters/Serverledge_adapter.hpp>
#include <ff/FaaS/ff_faas_function.hpp>
#include <iostream>

struct MyInput {
    int a;
    int b;
    /*
    static MyInput* faas_alloc(char* buffer, size_t) {
        return reinterpret_cast<MyInput*>(buffer);    
    }

    bool faas_deserialize(char*, size_t ) {
        return false;    
    }
    */
};

struct MyOutput {
    int result;
    /*
    std::tuple<char*, size_t, bool> faas_serialize() {
        return { reinterpret_cast<char*>(this), sizeof(int), false};
    }
    */
};


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


class MyFaaSFunction : public ff::ff_faas_function<MyInput, MyOutput, Serverledge_adapter> {
public:
    MyOutput* svc(MyInput* input) override {
        MyOutput* out = new MyOutput();
        cout << "Eseguendo la funzione FAAS con input: a = " << input->a << ", b = " << input->b << endl;
        out->result = input->a + input->b;
        cout << "Risultato della somma: " << out->result << endl;
        delete input;
        return out;
    }
};

int main(void) {
    MyFaaSFunction faasFunction;
    return faasFunction.run_and_wait_end();
}