#ifndef FF_FAAS_TYPETRAITS_HPP
#define FF_FAAS_TYPETRAITS_HPP

#include <type_traits>
#include <bitsery/bitsery.h>
#include <bitsery/adapter/measure_size.h>
#include <bitsery/details/adapter_common.h>

namespace ff {
    namespace traits {

        // Forward declarations (dichiarazioni, non definizioni)
        template <class T>
        bool faas_serialize(T& buffer, T* input);

        template <class T>
        bool faas_deserialize(const T& buffer, T*& output);

        template <class T>
        void faas_serializefreetask(char* data, T* obj);

        template <class T>
        void faas_deserializealloctask(const T& buffer, T*& obj);

        template <class T, template <class...> class Test>
        struct faas_exists{
            template<class U>
            static std::true_type check(Test<U>*);

            template<class U>
            static std::false_type check(...);

            static constexpr bool value = decltype(check<T>(0))::value;
        };

        template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(faas_serialize<std::pair<char*, size_t>>(std::declval<std::pair<char*, size_t>&>(), std::declval<U*>()))>>>
        struct user_serialize_faas_test{};

        template<typename T, typename = std::enable_if_t<traits::faas_exists<T, traits::user_serialize_faas_test>::value>>
        std::pair<char*,size_t> faas_serializeWrapper(T* in, bool& datacopied){
            std::pair<char*,size_t> p;
            datacopied = faas_serialize<std::pair<char*, size_t>>(p, in);
            return p;
        }

        template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(faas_deserialize<std::pair<char*, size_t>>(std::declval<const std::pair<char*, size_t>&>(), std::declval<U*&>()))>>>
        struct user_deserialize_faas_test{};

        template<typename T, typename = std::enable_if_t<traits::faas_exists<T, traits::user_deserialize_faas_test>::value>>
        bool faas_deserializeWrapper(char* c, size_t s, T*& obj){
            return faas_deserialize<std::pair<char*, size_t>>(std::make_pair(c, s),obj);
        }

        template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(faas_serializefreetask<char>(std::declval<char*>(), std::declval<U*>()))>>>
        struct user_freetask_faas_test{};

        template<typename T, typename = std::enable_if_t<traits::faas_exists<T, traits::user_freetask_faas_test>::value>>
        void faas_freetaskWrapper(T*in){
            faas_serializefreetask<char>((char*)in, in);
        }

        template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(faas_deserializealloctask<std::pair<char*, size_t>>(std::declval<const std::pair<char*, size_t>&>(), std::declval<U*&>()))>>>
        struct user_alloctask_faas_test{};
        
        template<typename T, typename = std::enable_if_t<traits::faas_exists<T, traits::user_alloctask_faas_test>::value>>
        void faas_alloctaskWrapper(char* c, size_t s, T*& p){
            faas_deserializealloctask<std::pair<char*, size_t>>(std::make_pair(c,s), p);
        }
            
        template<class U, class = std::enable_if_t<std::is_same_v<std::pair<char*,size_t>, decltype(faas_serializeWrapper<U>(std::declval<U*>(), std::declval<bool&>()))>>>
        struct serialize_faas_test{};

        template<class U, class = std::enable_if_t<std::is_same_v<bool, decltype(faas_deserializeWrapper<U>(std::declval<char*>(), std::declval<size_t&>(), std::declval<U*&>()))>>>
        struct deserialize_faas_test{};

        template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(faas_freetaskWrapper<U>(std::declval<U*>()))>>>
        struct freetask_faas_test{};

        template<class U, class = std::enable_if_t<std::is_same_v<void, decltype(faas_alloctaskWrapper<U>(std::declval<char*>(), std::declval<size_t&>(), std::declval<U*&>()))>>>
        struct alloctask_faas_test{};  

        /*
            High level traits to use
        */

        template<class T>
        using is_faas_serializable = faas_exists<T, serialize_faas_test>;

        //helper 
        template<class T>
        inline constexpr bool is_faas_serializable_v = is_faas_serializable<T>::value;


        template<class T>
        using is_faas_deserializable = faas_exists<T, deserialize_faas_test>;

        // helper
        template<class T>
        inline constexpr bool is_faas_deserializable_v = is_faas_deserializable<T>::value;

        template<class T>
        using has_faas_freetask = faas_exists<T, freetask_faas_test>;

        template<class T>
        inline constexpr bool has_faas_freetask_v = has_faas_freetask<T>::value;

        template<class T>
        using has_faas_alloctask = faas_exists<T, alloctask_faas_test>;

        template<class T>
        inline constexpr bool has_faas_alloctask_v = has_faas_alloctask<T>::value;
    }
} 

#endif // FF_FAAS_TYPETRAITS_HPP