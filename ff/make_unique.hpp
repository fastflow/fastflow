#ifndef FF_MAKEUNIQUE_HPP
#define FF_MAKEUNIQUE_HPP

#include <memory>
#include <type_traits>
#include <utility>


#if __cplusplus < 201400L   // to check
// C++11 implementation of make_unique

#if (__cplusplus >= 201103L) || (defined __GXX_EXPERIMENTAL_CXX0X__) || (defined(HAS_CXX11_VARIADIC_TEMPLATES))
template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(std::false_type, Args&&... args) {
    return std::unique_ptr<T>(new T(std::forward<Args>(args)...));
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique_helper(std::true_type, Args&&... args) {
   static_assert(std::extent<T>::value == 0,
       "make_unique<T[N]>() is forbidden, please use make_unique<T[]>().");

   typedef typename std::remove_extent<T>::type U;
   return std::unique_ptr<T>(new U[sizeof...(Args)]{std::forward<Args>(args)...});
}

template <typename T, typename... Args>
std::unique_ptr<T> make_unique(Args&&... args) {
   return make_unique_helper<T>(std::is_array<T>(), std::forward<Args>(args)...);
}
#endif

#else

using std::make_unique;// need to check if only the name is sufficient

#endif


#endif // FF_MAKEUNIQUE_HPP
