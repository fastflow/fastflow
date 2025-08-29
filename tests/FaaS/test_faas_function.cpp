#include <ff/FaaS/adapters/Serverledge_adapter.hpp>
#include <ff/FaaS/ff_faas_function.hpp>
#include <iostream>

struct MyInput {
    int a;
    int b;
};

struct MyOutput {
    int result;
};

template <class T>
bool faas_serialize(T& b, MyOutput* output) {
    b = {reinterpret_cast<char*>(output), sizeof(MyOutput)};
    return false;
}

template <class T>
bool faas_deserialize(const T& b, MyInput*& strPtr){
    strPtr = new (b.first) MyInput;
    return false;
}

template<typename T>
void faas_deserializealloctask(const T&, MyInput*&) {
}

/*
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
*/

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