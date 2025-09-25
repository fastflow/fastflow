#ifndef FF_FAAS_TYPETRAITS_HPP
#define FF_FAAS_TYPETRAITS_HPP

#include <type_traits>

namespace ff {
    namespace traits {
        template <typename T>                                                   
        struct has_faas_allocTask_member {                                                           
            template <typename U>                                               
            static auto test(U*) -> std::is_same<U*, decltype(std::declval<U>().faas_alloc(std::declval<char*>(), std::declval<size_t>()))>;
                                                                                
            template <typename>                                              
            static std::false_type test(...);                                   
                                                                                
            static constexpr bool value = decltype(test<T>(nullptr))::value; 
        };

        template <typename T>                                                   
        struct has_faas_deserialize_member {                                                           
            template <typename U>                                               
            static auto test(U*) -> std::is_same<bool, decltype(std::declval<U>().faas_deserialize(std::declval<char*>(), std::declval<size_t>()))>;
                                                                                
            template <typename>                                              
            static std::false_type test(...);                                   
                                                                                
            static constexpr bool value = decltype(test<T>(nullptr))::value; 
        };

        template <typename T>                                                   
        struct has_faas_serialize_member {                                                           
            template <typename U>    
            static auto test(U*) -> std::is_same<std::tuple<char*,size_t,bool>, decltype(std::declval<U>().faas_serialize())>;                                          
                                                                                
            template <typename>                                              
            static std::false_type test(...);                                   
                                                                                
            static constexpr bool value = decltype(test<T>(nullptr))::value; 
        };

        template <typename T>                                                   
        struct has_faas_freeTask_member {                                                           
            template <typename U>    
            static auto test(U*) -> std::is_same<void, decltype(std::declval<U>().faas_freeTask(std::declval<U*>()))>;                                          
                                                                                
            template <typename>                                              
            static std::false_type test(...);                                   
                                                                                
            static constexpr bool value = decltype(test<T>(nullptr))::value; 
        };

        template <typename T>                                                   
        struct has_faas_freeBlob_member {                                                           
            template <typename U>    
            static auto test(U*) -> std::is_same<void, decltype(std::declval<U>().faas_freeBlob(std::declval<char*>(), std::declval<size_t>()))>;                                          
                                                                                
            template <typename>                                              
            static std::false_type test(...);                                   
                                                                                
            static constexpr bool value = decltype(test<T>(nullptr))::value; 
        };
    }
} // namespace
#endif // FF_FAAS_TYPETRAITS_HPP
