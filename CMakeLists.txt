#
# If the user specifies -DCMAKE_BUILD_TYPE on the command line, take
# their definition and dump it in the cache along with proper
# documentation, otherwise set  CMAKE_BUILD_TYPE to Debug prior to
# calling PROJECT() 
#

IF(DEFINED CMAKE_BUILD_TYPE)
   SET(CMAKE_BUILD_TYPE ${CMAKE_BUILD_TYPE} CACHE STRING "Choose the type of
build, options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug
Release RelWithDebInfo MinSizeRel.")
   MESSAGE(STATUS "Build Type from command line " ${CMAKE_BUILD_TYPE})
ELSE()
   SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
   MESSAGE(STATUS "Default Build Type: Release  (change with
   -DCMAKE_BUILD_TYPE=Debug | Release | RelWithDebInfo | MinSizeRel")
ENDIF()


project( mc-fastflow )

cmake_minimum_required( VERSION 2.4 )

#if (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))
#  MESSAGE(STATUS "[Unix-like system with pthread]")
#else (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))
#  MESSAGE(STATUS "[Windows system: using Windows native threads]")
#endif (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))

find_package(Threads)
if (NOT(CMAKE_HAVE_PTHREAD_H))
  if (CMAKE_SYSTEM_NAME MATCHES "Windows")
     ADD_DEFINITIONS(-D_FF_SYSTEM_HAVE_WIN_PTHREAD)
    message(STATUS "Using FF Pthread minport")
    set(FFHEADERS_AT
      ${FF}/platforms/platform.h
      ${FF}/platforms/pthread_minport_windows.h
      ${FF}/platforms/platform_msvc_windows.h
    )
  else (CMAKE_SYSTEM_NAME MATCHES "Windows")
     message(FATAL_ERROR "Cannot find Pthreads")
  endif  (CMAKE_SYSTEM_NAME MATCHES "Windows")
endif (NOT(CMAKE_HAVE_PTHREAD_H))

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")

set( FF ${PROJECT_SOURCE_DIR}/ff/ )
set(FFHEADERS
    ${FF}/allocator.hpp
    ${FF}/buffer.hpp
    ${FF}/cycle.h
    ${FF}/farm.hpp
    ${FF}/gt.hpp
    ${FF}/lb.hpp
    ${FF}/node.hpp
    ${FF}/pipeline.hpp
    ${FF}/spin-lock.hpp
    ${FF}/squeue.hpp
    ${FF}/svector.hpp
    ${FF}/sysdep.h
    ${FF}/ubuffer.hpp
    ${FF}/utils.hpp
    ${FF}/version.h
    ${FF}/dynqueue.hpp
    ${FF}/mapping_utils.hpp
    ${FF}/MPMCqueues.hpp
   ) 

set(FFHEADERS_AT
    ${FF}/atomic/atomic.h
    ${FF}/atomic/atomic-i386.h
    ${FF}/atomic/atomic-ppc.h
    ${FF}/atomic/atomic-x86_64.h
    ${FF}/atomic/abstraction_dcas.h
    ${FF}/platforms/platform.h
   )

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
option( BUILD_EXAMPLES      "Build examples"        ON )
MESSAGE(STATUS "Configuring >>examples<< directory")
else (CMAKE_SYSTEM_NAME MATCHES "Windows")
option( BUILD_EXAMPLES      "Build examples"        ON )
MESSAGE(STATUS "Configuring >>examples<< directory")
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")

#       NAME                "Description"           DEFAULT
option( BUILD_TESTS         "Build tests"           ON ) 
MESSAGE(STATUS "Configuring >>tests<< directory")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY )

add_custom_target( uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" )

install( FILES ${FFHEADERS} DESTINATION /include/ff/ )
install( FILES ${FFHEADERS_AT} DESTINATION /include/ff/atomic )

if( BUILD_EXAMPLES )
    enable_testing()
    add_subdirectory( examples )
endif( BUILD_EXAMPLES )

if( BUILD_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif( BUILD_TESTS )
