/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*!
 *  \file ff_faas_function_adapter.hpp
 *  \ingroup adapter
 *
 *  \brief  Defines the ff_faas_function_adapter class for a specific FAAS framework
 *
 *  It contains the definition of the \p ff_faas_function_adapter class,
 *  with features oriented offloading to FAAS systems.
 *
 */

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

#ifndef FF_FAAS_FUNCTION_ADAPTER_HPP
#define FF_FAAS_FUNCTION_ADAPTER_HPP

#include <ff/FaaS/ff_faas_buffer.hpp>
#include <memory>
namespace adapter {
    class ff_faas_function_adapter
    {
        public:
            ff_faas_function_adapter() = default;

            virtual ~ff_faas_function_adapter() = default;

            virtual bool ff_faas_function_adapter_init() = 0;
            virtual std::unique_ptr<ff::faasBuffer> ff_faas_function_adapter_getRequest(std::unique_ptr<std::string>&, bool&) = 0; // The first parameter is an error messag, the second parameter indicates if the function execution time must be measured
            virtual bool ff_faas_function_adapter_sendResponse(std::unique_ptr<ff::faasBuffer>,std::unique_ptr<std::string>&, double) = 0; // The first parameter is an error message, the second parameter is the function execution time, in microseconds, or -1 if not available
            virtual bool ff_faas_function_adapter_sendErrorResponse(std::unique_ptr<std::string>) = 0;

    };
}

#endif // FF_FAAS_FUNCTION_ADAPTER_HPP