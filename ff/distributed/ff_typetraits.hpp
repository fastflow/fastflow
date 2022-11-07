#ifndef FF_TYPETRAITS_H
#define FF_TYPETRAITS_H

#include <type_traits>

namespace ff{

namespace traits {

    using yes = std::true_type;
    using no  = std::false_type;

    template <class T, class A>                                                                                             
      struct has_non_member_serialize_impl                                                                                  
      {                                                                                                                         
        template <class TT, class AA>                                                                                           
        static auto test(int) -> decltype( serialize(std::declval<AA&>(), std::declval<TT*>()), yes());                  
        template <class, class>                                                                                                 
        static no test( ... );                                                                                                  
        static const bool value = std::is_same<decltype( test<T, A>( 0 ) ), yes>::value;                                        
      };

    template <class T>                                                                                                 
    struct has_non_member_serialize : std::integral_constant<bool, has_non_member_serialize_impl<T, std::pair<char*, size_t>>::value> {};


    template <class T, class A>                                                                                             
      struct has_non_member_deserialize_impl                                                                                  
      {                                                                                                                         
        template <class TT, class AA>                                                                                           
        static auto test(int) -> decltype( deserialize(std::declval<AA&>(), std::declval<TT*>()), yes());                  
        template <class, class>                                                                                                 
        static no test( ... );                                                                                                  
        static const bool value = std::is_same<decltype( test<T, A>( 0 ) ), yes>::value;                                        
      };

    template <class T>                                                                                                 
    struct has_non_member_deserialize : std::integral_constant<bool, has_non_member_deserialize_impl<T, const std::pair<char*, size_t>>::value> {};

    template <class T, class A>                                                                                             
      struct has_non_member_serializefreetask_impl                                                                                  
      {                                                                                                                         
        template <class TT, class AA>                                                                                           
        static auto test(int) -> decltype( serializefreetask(std::declval<AA*>(), std::declval<TT*>()), yes());                  
        template <class, class>                                                                                                 
        static no test( ... );                                                                                                  
        static const bool value = std::is_same<decltype( test<T, A>( 0 ) ), yes>::value;                                        
      };

    template <class T>                                                                                                 
    struct has_non_member_serializefreetask : std::integral_constant<bool, has_non_member_serializefreetask_impl<T, char>::value> {};


    template <class T, class A>                                                                                             
      struct has_non_member_deserializealloctask_impl                                                                                  
      {                                                                                                                         
        template <class TT, class AA>                                                                                           
        static auto test(int) -> decltype( deserializealloctask(std::declval<AA&>(), std::declval<TT*&>()), yes());                  
        template <class, class>                                                                                                 
        static no test( ... );                                                                                                  
        static const bool value = std::is_same<decltype( test<T, A>( 0 ) ), yes>::value;                                        
      };

    template <class T>                                                                                                 
    struct has_non_member_deserializealloctask : std::integral_constant<bool, has_non_member_deserializealloctask_impl<T, const std::pair<char*, size_t>>::value> {};

template<class T>
using is_serializable = has_non_member_serialize<T>;

//helper 
template<class T>
inline constexpr bool is_serializable_v = is_serializable<T>::value;


template<class T>
using is_deserializable = has_non_member_deserialize<T>;

// helper
template<class T>
inline constexpr bool is_deserializable_v = is_deserializable<T>::value;

template<class T>
using has_freetask = has_non_member_serializefreetask<T>;

template<class T>
inline constexpr bool has_freetask_v = has_freetask<T>::value;

template<class T>
using has_alloctask = has_non_member_deserializealloctask<T>;

template<class T>
inline constexpr bool has_alloctask_v = has_alloctask<T>::value;

}

template<typename T, typename = std::enable_if_t<traits::is_serializable<T>::value>>
std::pair<char*,size_t> serializeWrapper(T*in, bool& datacopied){
    std::pair<char*,size_t> p;
    datacopied = serialize(p, in);
    return p;
}

template<typename T, typename = std::enable_if_t<traits::is_deserializable<T>::value>>
bool deserializeWrapper(char* c, size_t s, T* obj){
    return deserialize(std::make_pair(c, s),obj);
}

template<typename T, typename = std::enable_if_t<traits::has_freetask<T>::value>>
void freetaskWrapper(T*in){
    serializefreetask((char*)in, in);
}

template<typename T, typename = std::enable_if_t<traits::has_alloctask<T>::value>>
void alloctaskWrapper(char* c, size_t s, T*& p){
    deserializealloctask(std::make_pair(c,s), p);
}

} // namespace
#endif
