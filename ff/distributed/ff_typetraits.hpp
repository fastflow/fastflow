/* ***************************************************************************
 *
 *  FastFlow is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *  Starting from version 3.0.1 FastFlow is dual licensed under the GNU LGPLv3
 *  or MIT License (https://github.com/ParaGroup/WindFlow/blob/vers3.x/LICENSE.MIT)
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 *
 * Authors: 
 *   Nicolo' Tonci
 *   Massimo Torquati
 */

#ifndef FF_TYPETRAITS_H
#define FF_TYPETRAITS_H

#include <type_traits>

namespace ff {

namespace traits {

  template <typename T>                                                   
  struct has_allocTask_member {                                                           
      template <typename U>                                               
      static auto test(U*) -> std::is_same<U*, decltype(std::declval<U>().alloc(std::declval<char*>(), std::declval<size_t>()))>;
                                                                          
      template <typename>                                              
      static std::false_type test(...);                                   
                                                                          
      static constexpr bool value = decltype(test<T>(nullptr))::value; 
  };

  template <typename T>                                                   
  struct has_deserialize_member {                                                           
      template <typename U>                                               
      static auto test(U*) -> std::is_same<bool, decltype(std::declval<U>().deserialize(std::declval<char*>(), std::declval<size_t>()))>;
                                                                          
      template <typename>                                              
      static std::false_type test(...);                                   
                                                                          
      static constexpr bool value = decltype(test<T>(nullptr))::value; 
  };

  template <typename T>                                                   
  struct has_serialize_member {                                                           
      template <typename U>    
      static auto test(U*) -> std::is_same<std::tuple<char*,size_t,bool>, decltype(std::declval<U>().serialize())>;                                          
                                                                          
      template <typename>                                              
      static std::false_type test(...);                                   
                                                                          
      static constexpr bool value = decltype(test<T>(nullptr))::value; 
  };

  template <typename T>                                                   
  struct has_freeTask_member {                                                           
      template <typename U>    
      static auto test(U*) -> std::is_same<void, decltype(std::declval<U>().freeTask(std::declval<U*>()))>;                                          
                                                                          
      template <typename>                                              
      static std::false_type test(...);                                   
                                                                          
      static constexpr bool value = decltype(test<T>(nullptr))::value; 
  };

  template <typename T>                                                   
  struct has_freeBlob_member {                                                           
      template <typename U>    
      static auto test(U*) -> std::is_same<void, decltype(std::declval<U>().freeBlob(std::declval<char*>(), std::declval<size_t>()))>;                                          
                                                                          
      template <typename>                                              
      static std::false_type test(...);                                   
                                                                          
      static constexpr bool value = decltype(test<T>(nullptr))::value; 
  };

}
} // namespace
#endif
