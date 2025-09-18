#include "ff/ff_faas.hpp"
#include <bitsery/bitsery.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/traits/vector.h>
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <simdutf/simdutf.h>
#include <simdutf/simdutf.cpp>
#include <iostream>
#include <vector>

#ifdef DEBUG
std::mutex debug_output_mutex;
#endif

struct MyInput {
    int a, b;
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
    strPtr = new (b.first) MyOutput;
    return false;
}

template<typename T>
void faas_deserializealloctask(const T&, MyOutput*&) { 
}

/*
// Bitsery
template <typename S>
void serialize(S& s, MyInput& obj) {
    s.value4b(obj.a);
    s.value4b(obj.b);
}

template <typename S>
void serialize(S& s, MyOutput& obj) {
    s.value4b(obj.result);
}
*/

int main(int argc, char* argv[]) {
    // 1. Prepara l'input
    MyInput input{5, 7};
    bool stats_collection = false;
    if(argc==2)
        if(strcmp(argv[1],"t")==0)
            stats_collection = true;
    size_t buffer_size = 0;
    char* buffer = nullptr;


    PRINT_DBG("Inizio preparazione dell'input...");
    PRINT_DBG("Input preparato: a = " + std::to_string(input.a) + ", b = " + std::to_string(input.b));

    if constexpr (ff::traits::is_faas_serializable_v<MyInput>) {
        bool datacopied = true;
        PRINT_DBG("Inizio serializzazione manuale...");
        std::pair<char*, size_t> p = ff::traits::faas_serializeWrapper<MyInput>(&input, datacopied);
        buffer_size = p.second;
        buffer = p.first;
        PRINT_DBG("Serializzazione manuale completata, buffer_size: " + std::to_string(buffer_size));
    }
    else {
        // 2. Serializza in un buffer binario (std::vector<char>)
        PRINT_DBG("Inizio serializzazione con Bitsery...");
        bitsery::MeasureSize measureSize;
        size_t neededSize = bitsery::quickSerialization<bitsery::MeasureSize>(measureSize, input);
        PRINT_DBG("Dimensione necessaria per la serializzazione: " + std::to_string(neededSize));

        std::vector<char> Bitsery_buffer;
        Bitsery_buffer.resize(neededSize);
        PRINT_DBG("Buffer ridimensionato a " + std::to_string(neededSize) + " byte");

        bitsery::quickSerialization<bitsery::OutputBufferAdapter<std::vector<char>>>(Bitsery_buffer, input);
        PRINT_DBG("Serializzazione con Bitsery completata");
        buffer_size = Bitsery_buffer.size();
        buffer = Bitsery_buffer.data(); // Converti in un array di char
    }

    // 3. Codifica in Base64 con fastbase64
    PRINT_DBG("Inizio codifica in Base64...");

    
    std::vector<char> bufferBase64;
    bufferBase64.resize(simdutf::base64_length_from_binary(buffer_size, simdutf::base64_options::base64_default));

    simdutf::binary_to_base64(buffer, buffer_size, bufferBase64.data(), simdutf::base64_options::base64_default);
    PRINT_DBG("Codifica Base64 completata, lunghezza: " + std::to_string(bufferBase64.size()));

    // 4. Costruisci JSON con RapidJSON
    PRINT_DBG("Creazione JSON...");
    rapidjson::Document doc;
    doc.SetObject();
    rapidjson::Document::AllocatorType& allocator = doc.GetAllocator();

    rapidjson::Value commandArray(rapidjson::kArrayType);
    commandArray.PushBack("testfaasfunction", allocator);
    doc.AddMember("Command", commandArray, allocator);

    rapidjson::Value paramsObj(rapidjson::kObjectType);
    rapidjson::Value base64Val;
    base64Val.SetString(bufferBase64.data(), static_cast<rapidjson::SizeType>(bufferBase64.size()), allocator);
    paramsObj.AddMember("p", base64Val, allocator);
    if (stats_collection)
        paramsObj.AddMember("s","",allocator);
    doc.AddMember("Params", paramsObj, allocator);

    doc.AddMember("Handler", "", allocator);
    doc.AddMember("HandlerDir", "", allocator);
    doc.AddMember("ReturnOutput", true, allocator);

    rapidjson::StringBuffer jsonBuf;
    rapidjson::Writer<rapidjson::StringBuffer> writer(jsonBuf);
    doc.Accept(writer);
    std::string jsonBody = jsonBuf.GetString();
    PRINT_DBG("JSON creato: " + jsonBody);

    // 5. Crea il client HTTP
    PRINT_DBG("Inizio richiesta HTTP...");
    httplib::Client cli("0.0.0.0", 8080);
    cli.set_connection_timeout(5);

    auto res = cli.Post("/invoke", jsonBody, "application/json");
    if (!res || res->status != 200) {
        std::cerr << "Errore HTTP o nessuna risposta. Status: " << (res ? std::to_string(res->status) : "n/a\n");
        return 1;
    }

    PRINT_DBG("Risposta HTTP ricevuta, status: " + std::to_string(res->status));
    std::vector<char> respBuffer(res->body.begin(), res->body.end());

    PRINT_DBG("Risposta HTTP ricevuta: " + std::string(respBuffer.data()));

    // Parsing del corpo JSON della risposta
    rapidjson::Document responseDoc;
    responseDoc.Parse(respBuffer.data(), respBuffer.size());

    if (responseDoc.HasParseError()) {
        std::cerr << "Errore nel parsing della risposta JSON.\n";
        return 1;
    }

    rapidjson::Value* p; 
    if (responseDoc.HasMember("result") && responseDoc["result"].IsString())
        p = &responseDoc["result"];
    else {
        std::cerr << "Errore: la risposta non contiene il campo 'result', o non è del tipo giusto\n";
        return 1;
    }
    
    responseDoc.Parse(p->GetString(), p->GetStringLength()); 
    if (responseDoc.HasParseError()) {
        std::cerr << "Errore nel parsing della risposta JSON.\n";
        return 1;
    }

    if(!responseDoc.HasMember("r") || !responseDoc["r"].IsString()) {
        std::cerr << "Errore: la risposta non contiene il campo 'r' o non è del tipo stringa\n";
        return 1;
    }

    p =  &responseDoc["r"];

    size_t rSize = p->GetStringLength();
    size_t maxLength = simdutf::maximal_binary_length_from_base64(p->GetString(), rSize);
    PRINT_DBG("Decoding Base64, input size: " + std::to_string(rSize) + ", maxLength: " + std::to_string(maxLength));

    char* resultData = new char[maxLength];
    simdutf::result Res = simdutf::base64_to_binary(p->GetString(), rSize, resultData, simdutf::base64_default, simdutf::last_chunk_handling_options::strict);
    if(Res.error) {
        std::cerr << "Internal error: conversion from Base64 to binary for the input parameters failed\n";
        PRINT_DBG("Base64 to binary conversion failed.");
        return 1;
    }

    MyOutput* output;

    if constexpr (ff::traits::is_faas_deserializable_v<MyOutput>) {
        PRINT_DBG("Inizio deserializzazione manuale...");
        ff::traits::faas_alloctaskWrapper<MyOutput>(resultData, maxLength, output);
        ff::traits::faas_deserializeWrapper<MyOutput>(resultData, maxLength, output);
        PRINT_DBG("Deserializzazione manuale completata");
    }
    else {
        // 6. Deserializza la risposta (binaria)
        std::vector<char> v = std::vector<char>(resultData, resultData + Res.count);
        output = new MyOutput;
        PRINT_DBG("Inizio deserializzazione della risposta con Bitsery...");
        auto state = bitsery::quickDeserialization<bitsery::InputBufferAdapter<std::vector<char>>>(
            {v.begin(), v.end()}, output
        );

        if (state.first != bitsery::ReaderError::NoError) {
            std::cerr << "Errore nella deserializzazione\n";
            return 1;
        }
        PRINT_DBG("Deserializzazione con Bitsery completata");
    }
    std::cout << "Risultato della somma: " << output->result << "\n";

    return 0;
}