#include "ff/ff_faas.hpp"
#include <ff/FaaS/ff_faas_buffer.hpp>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/adapter/buffer.h>
#include <bitsery/adapter/stream.h>
#include <httplib.h>
#include <rapidjson/document.h>
#include <rapidjson/writer.h>
#include <rapidjson/stringbuffer.h>
#include <simdutf/simdutf.h>
#include <simdutf/simdutf.cpp>
#include <iostream>
#include <vector>
#include <tuple>

#ifdef DEBUG
std::mutex debug_output_mutex;
#endif

struct MyInput {
    int a, b;

    //std::tuple<char*, size_t, bool> faas_serialize() {
    //    return {reinterpret_cast<char*>(this), sizeof(MyInput), false};
    //}
        
};

struct MyOutput {
    int result;

    //static MyOutput* faas_alloc(char* buffer, size_t sz) {
    //    return reinterpret_cast<MyOutput*>(buffer);    
    //} 

    //bool faas_deserialize(char*, size_t ) {
    //    return false;    
    //}
};


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

template <typename T>
std::enable_if_t<ff::traits::has_faas_serialize_member<T>::value>
manual_serialize(T& input, char*& buffer, size_t& buffer_size) {
        PRINT_DBG("Inizio serializzazione manuale...");
        std::tuple<char*, size_t, bool> p = input.faas_serialize();
        buffer_size = std::get<1>(p);
        buffer = std::get<0>(p);
        PRINT_DBG("Serializzazione manuale completata, buffer_size: " + std::to_string(buffer_size));
}

template <typename T>
std::enable_if_t<!ff::traits::has_faas_serialize_member<T>::value>
manual_serialize(T& input, char*& buffer, size_t& buffer_size) {
    PRINT_DBG("Inizio serializzazione con Bitsery...");
    bitsery::MeasureSize measureSize;
    buffer_size = bitsery::quickSerialization<bitsery::MeasureSize>(measureSize, input);
    PRINT_DBG("Dimensione necessaria per la serializzazione: " + std::to_string(buffer_size));

    buffer = new char[buffer_size];

    ff::faasBuffer faas_buffer;
    faas_buffer.setBuffer(buffer, buffer_size, false);

    PRINT_DBG("Buffer dimensionato a " + std::to_string(buffer_size) + " byte");

    std::ostream os(&faas_buffer);           
    bitsery::Serializer<bitsery::OutputStreamAdapter> ser(os);
    ser.object(input);  
    PRINT_DBG("Serializzazione con Bitsery completata");
}

template <typename T>
std::enable_if_t<ff::traits::has_faas_deserialize_member<T>::value>
manual_deserialize(T*& output, char*& resultData, size_t& maxLength) {
        PRINT_DBG("Inizio deserializzazione manuale...");
        output = T::faas_alloc(resultData, maxLength);
        output->faas_deserialize(resultData, maxLength);
        PRINT_DBG("Deserializzazione manuale completata");
}

template <typename T>
std::enable_if_t<!ff::traits::has_faas_deserialize_member<T>::value>
manual_deserialize(T*& output, char*& resultData, size_t& maxLength) {
    PRINT_DBG("Inizio deserializzazione della risposta con Bitsery...");
    ff::faasBuffer faas_buffer;
    faas_buffer.setBuffer(resultData, maxLength);
    output = new T;
    std::istream is(&faas_buffer);
    bitsery::Deserializer<bitsery::InputStreamAdapter> des(is);
    des.object(*output);
    PRINT_DBG("Deserializzazione con Bitsery completata");
}

int main() {
    MyInput input{5, 10};
    char* buffer = nullptr;
    size_t buffer_size = 0;
    bool stats_collection = false;

    manual_serialize<MyInput>(input, buffer, buffer_size);

    PRINT_DBG("Inizio codifica in Base64...");

    std::vector<char> bufferBase64;
    bufferBase64.resize(simdutf::base64_length_from_binary(buffer_size, simdutf::base64_options::base64_default));

    simdutf::binary_to_base64(buffer, buffer_size, bufferBase64.data(), simdutf::base64_options::base64_default);
    PRINT_DBG("Codifica Base64 completata, lunghezza: " + std::to_string(bufferBase64.size()));

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
    manual_deserialize<MyOutput>(output, resultData, maxLength);

    std::cout << "Risultato della somma: " << output->result << "\n";
    return 0;
}