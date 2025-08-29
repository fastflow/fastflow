/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */


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
 */

/* Author: Luca Ferrucci
 *
 */

// This file contains some configuration variables for the FaaS support
//

#ifndef FF_FAAS_CONFIGURATION_HPP
#define FF_FAAS_CONFIGURATION_HPP

// default size of the default parallelism degree
#if !defined(DEFAULT_PARALLELISM_DEGREE)
#define DEFAULT_PARALLELISM_DEGREE 1
#endif

// default filename for the FAAS backends file
#if !defined(DEFAULT_BACKEND_FILE)
#define DEFAULT_BACKEND_FILE "faas_backends.json"
#endif

// default filename for the FAAS functions file
#if !defined(DEFAULT_FUNCTIONS_FILE)
#define DEFAULT_FUNCTIONS_FILE "faas_functions.json"
#endif

#endif /* FF_CONFIG_HPP */