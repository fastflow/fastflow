# - Try to find ZMQ
# Once done this will define
#
#  ZMQ_FOUND - system has ZMQ
#  ZMQ_INCLUDE_DIRS - the ZMQ include directory
#  ZMQ_LIBRARIES - Link these to use ZMQ
#  ZMQ_DEFINITIONS - Compiler switches required for using ZMQ
#
#  Copyright (c) 2011 Lee Hambley <lee.hambley@gmail.com>
#
#  Redistribution and use is allowed according to the terms of the New
#  BSD license.
#  For details see the accompanying COPYING-CMAKE-SCRIPTS file.
#

if (ZMQ_LIBRARIES AND ZMQ_INCLUDE_DIRS)
  # in cache already
  set(ZMQ_FOUND TRUE)
else (ZMQ_LIBRARIES AND ZMQ_INCLUDE_DIRS)
  find_path(ZMQ_INCLUDE_DIR
    NAMES
      zmq.h
    PATHS
      /usr/include
      /usr/local/include
      /opt/local/include
      /sw/include
  )

  find_library(ZMQ_LIBRARY
    NAMES
      zmq
    PATHS
      /usr/lib
      /usr/local/lib
      /opt/local/lib
      /sw/lib
  )

  set(ZMQ_INCLUDE_DIRS
    ${ZMQ_INCLUDE_DIR}
  )

  if (ZMQ_LIBRARY)
    set(ZMQ_LIBRARIES
        ${ZMQ_LIBRARIES}
        ${ZMQ_LIBRARY}
    )
  endif (ZMQ_LIBRARY)

  include(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(ZMQ DEFAULT_MSG ZMQ_LIBRARIES ZMQ_INCLUDE_DIRS)

  # show the ZMQ_INCLUDE_DIRS and ZMQ_LIBRARIES variables only in the advanced view
  mark_as_advanced(ZMQ_INCLUDE_DIRS ZMQ_LIBRARIES)

endif (ZMQ_LIBRARIES AND ZMQ_INCLUDE_DIRS)

if ( (ZMQ_LIBRARIES) AND (ZMQ_INCLUDE_DIRS) )
message(STATUS ${ZMQ_LIBRARIES})
set(ZMQ_FOUND TRUE)
endif ( (ZMQ_LIBRARIES) AND (ZMQ_INCLUDE_DIRS) )

### Alternative code

# - Try to find libzmq
# Once done, this will define
#
#  ZeroMQ_FOUND - system has libzmq
#  ZeroMQ_INCLUDE_DIRS - the libzmq include directories
#  ZeroMQ_LIBRARIES - link these to use libzmq

#include(LibFindMacros)

#IF (UNIX)
#	# Use pkg-config to get hints about paths
#	libfind_pkg_check_modules(ZeroMQ_PKGCONF libzmq)
#
#	# Include dir
# 	find_path(ZeroMQ_INCLUDE_DIR
# 	  NAMES zmq.hpp
# 	  PATHS ${ZEROMQ_ROOT}/include ${ZeroMQ_PKGCONF_INCLUDE_DIRS}
# 	)

# 	# Finally the library itself
# 	find_library(ZeroMQ_LIBRARY
# 	  NAMES zmq
# 	  PATHS ${ZEROMQ_ROOT}/lib ${ZeroMQ_PKGCONF_LIBRARY_DIRS}
# 	)
# ELSEIF (WIN32)
# 	message(STATUS "Checking 0mq for windows")
# 	find_path(ZeroMQ_INCLUDE_DIR
# 	  NAMES zmq.hpp
# 	  PATHS ${ZEROMQ_ROOT}/include ${CMAKE_INCLUDE_PATH}
# 	)
# 	message(STATUS "0mq ${ZeroMQ_INCLUDE_DIR}" )

# 	# Finally the library itself
# 	find_library(ZeroMQ_LIBRARY
# 	  NAMES zmq libzmq-v100-mt
# 	  PATHS ${ZEROMQ_ROOT}/lib ${CMAKE_LIB_PATH}
# 	)
# 	message(STATUS "0mq ${ZeroMQ_LIBRARY}" )
# ENDIF()

# # Set the include dir variables and the libraries and let libfind_process do the rest.
# # NOTE: Singular variables for this library, plural for libraries this this lib depends on.
# set(ZeroMQ_PROCESS_INCLUDES ZeroMQ_INCLUDE_DIR ZeroMQ_INCLUDE_DIRS)
# set(ZeroMQ_PROCESS_LIBS ZeroMQ_LIBRARY ZeroMQ_LIBRARIES)
# libfind_process(ZeroMQ)