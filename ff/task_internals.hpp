/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 
 *  \link
 *  \file task_internals.hpp
 *  \ingroup aux_classes
 *
 *  \brief Internal classes and helping functions for tasks management.
 */

#ifndef FF_TASK_INTERNALS_HPP
#define FF_TASK_INTERNALS_HPP
 
/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
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
 */

/* 
 * Author: Massimo Torquati (September 2014)
 *
 *
 */
#include <functional>
#include <tuple>
#include <vector>


namespace ff {

/* -------------- expanding tuple to func. arguments ------------------------- */
template<size_t N>
struct Apply {
    template<typename F, typename T, typename... A>
    static inline auto apply(F && f, T && t, A &&... a)
        -> decltype(Apply<N-1>::apply(
            ::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...))
    {
        return Apply<N-1>::apply(::std::forward<F>(f), ::std::forward<T>(t),
            ::std::get<N-1>(::std::forward<T>(t)), ::std::forward<A>(a)...
        );
    }
};

template<>
struct Apply<0> {
    template<typename F, typename T, typename... A>
    static inline auto apply(F && f, T &&, A &&... a)
        -> decltype(::std::forward<F>(f)(::std::forward<A>(a)...))
    {
        return ::std::forward<F>(f)(::std::forward<A>(a)...);
    }
};

template<typename F, typename T>
inline auto apply(F && f, T && t)
    -> decltype(Apply< ::std::tuple_size<typename ::std::decay<T>::type
					 >::value>::apply(::std::forward<F>(f), ::std::forward<T>(t)))
{
    return Apply< ::std::tuple_size<typename ::std::decay<T>::type>::value>::apply(::std::forward<F>(f), ::std::forward<T>(t));
}
/* --------------------------------------------------------------------------- */ 

/// kind of dependency
typedef enum {INPUT=0,OUTPUT=1,VALUE=2} data_direction_t;
/// generic pameter information (tag and kind of dependency)
struct param_info {
    uintptr_t        tag;  // unique tag for the parameter
    data_direction_t dir;
};
/// base class for a generic function call
struct base_f_t {
    virtual inline void call() {};
    virtual ~base_f_t() {};
};    
/// task function basic type
struct task_f_t { 
    std::vector<param_info> P;  // FIX: svector should be used here
    base_f_t *wtask;
};
/// task function
template<typename F_t, typename... Param>
struct ff_task_f_t: public base_f_t {
    ff_task_f_t(const F_t F, Param&... a):F(F) { args = std::make_tuple(a...);}	
    inline void call() { apply(F, args); }
    F_t F;
    std::tuple<Param...> args;	
};
    
} // namespace


#endif // FF_TASK_INTERNALS_HPP
