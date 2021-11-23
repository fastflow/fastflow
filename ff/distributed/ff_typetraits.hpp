#ifndef FF_TYPETRAITS_H
#define FF_TYPETRAITS_H

#include <type_traits>

namespace ff{

namespace traits {

template <class T, template <class...> class Test>
struct exists{
    template<class U>
    static std::true_type check(Test<U>*);

    template<class U>
    static std::false_type check(...);

    static constexpr bool value = decltype(check<T>(0))::value;
};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(serialize<std::pair<char*, size_t>>(std::declval<std::pair<char*, size_t>&>(), std::declval<U*>()))>>>
struct user_serialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(deserialize<std::pair<char*, size_t>>(std::declval<const std::pair<char*, size_t>&>(), std::declval<U*&>()))>>>
struct user_deserialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<std::pair<char*,size_t>, decltype(serializeWrapper<U>(std::declval<U*>()))>>>
struct serialize_test{};

template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(deserializeWrapper<U>(std::declval<char*>(), std::declval<size_t&>(), std::declval<U*&>()))>>>
struct deserialize_test{};



/*
    High level traits to use
*/

template<class T>
using is_serializable = exists<T, serialize_test>;

//helper 
template<class T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;


template<class T>
using is_deserializable = exists<T, deserialize_test>;

// helper
template<class T>
inline constexpr bool is_deserializable_v = is_deserializable<T>::value;

}


/*
    Wrapper to user defined serialize and de-serialize functions, in order to exploits user defined functions in other translation units. 
*/
template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_serialize_test>::value>>
std::pair<char*,size_t> serializeWrapper(T*in){
    std::pair<char*,size_t> p;
    serialize<std::pair<char*, size_t>>(p, in);
    return p;
}

template<typename T, typename = std::enable_if_t<traits::exists<T, traits::user_deserialize_test>::value>>
void deserializeWrapper(char* c, size_t s, T*& obj){
    deserialize<std::pair<char*, size_t>>(std::make_pair(c, s),obj);
}


}
#endif
