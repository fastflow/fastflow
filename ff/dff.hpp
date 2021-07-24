/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


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

#ifndef FF_DFF_HPP
#define FF_DFF_HPP

#define DFF_ENABLED
#if !defined(DFF_EXCLUDE_MPI)
#define DFF_MPI
#endif
#if !defined(DFF_EXCLUDE_BLOCKING)
#define BLOCKING_MODE
#else
#undef BLOCKING_MODE
#endif

#include <ff/ff.hpp>
#include <ff/distributed/ff_network.hpp>
#include <ff/distributed/ff_dgroups.hpp>
#include <ff/distributed/ff_dgroup.hpp>

#endif /* FF_DFF_HPP */
