/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
#ifndef _FF_PLATFORM_HPP_
#define _FF_PLATFORM_HPP_
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License version 3 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *
 ****************************************************************************
 */


#if (defined(_MSC_VER) || defined(__INTEL_COMPILER)) && defined(_WIN32)
#include "ff/platforms/platform_msvc_windows.h"
//#if (!defined(_FF_SYSTEM_HAVE_WIN_PTHREAD))
	#include "ff/platforms/pthread_minport_windows.h"
//#endif
#include<algorithm>
#elif defined(__GNUC__) && (defined(__linux) || defined(__APPLE__))
#include <sys/time.h>
#include <unistd.h>
#include <pthread.h>
//#define __TICKS2WAIT 1000
#else
#   error "unknown platform"
#endif

#endif //_FF_PLATFORM_HPP_


