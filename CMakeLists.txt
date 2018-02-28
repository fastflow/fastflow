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

cmake_minimum_required( VERSION 2.8 )

SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules )
FILE (COPY ${PROJECT_SOURCE_DIR}/cmake.modules/templates/ffconfig.h DESTINATION ${PROJECT_SOURCE_DIR}/cmake.modules/)

INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckCXXSourceCompiles.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)

message (STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR})
MESSAGE( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )
if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    MESSAGE( STATUS "C++ Compiler is Clang")
    SET(FFCM_HAS_CLANGXX "TRUE")
   # SET(CMAKE_CXX_FLAGS "-stdlib=libc++ -std=gnu++11")
   SET(CMAKE_CXX_FLAGS "-std=c++11")
   SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
#    ADD_DEFINITIONS(-stdlib=libc++ -std=gnu++11)  # Or -std=c++11
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    MESSAGE( STATUS "C++ Compiler is GNU")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GXX_VERSION)
    if (NOT(GXX_VERSION VERSION_GREATER 4.1))
       set(GCC_TOO_OLD 1)       
       message(WARNING "g++ Version < 4.1, some targets will be skipped")
       message(STATUS "Use CC/CXX env variable to define a newer compiler, e.g. export CC=/usr/bin/gcc-4.2; export CXX=/usr/bin/g++-4.2 ")
    else (NOT(GXX_VERSION VERSION_GREATER 4.1))
           ADD_DEFINITIONS(-std=c++11) # or -std=c++0x 
    endif  (NOT(GXX_VERSION VERSION_GREATER 4.1))    
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
     MESSAGE( STATUS "C++ Compiler is Intel")
     MESSAGE( WARNING "Compliance check not fully implemented yet")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
     MESSAGE( STATUS "C++ Compiler is Visual Studio")
endif()

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
  ADD_DEFINITIONS(-DNO_DEFAULT_MAPPING)
  message(WARNING "Pinning on Win platforn currently is not default" )
  if(CMAKE_CL_64)
    message(STATUS "64-bit microsoft compiler found")
    ADD_DEFINITIONS(-D_WIN64)
    ADD_DEFINITIONS(-D_AMD64_)
    ADD_DEFINITIONS(-DNOMINMAX)
#	ADD_DEFINITIONS(-D_WIN32_WINNT=0x0403)
  else(CMAKE_CL_64)
    message(STATUS "32-bit microsoft compiler found") 
#	ADD_DEFINITIONS(-D_WIN32_WINNT=0x0403)
  endif(CMAKE_CL_64)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")  
#if (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))
#  MESSAGE(STATUS "[Unix-like system with pthread]")
#else (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))
#  MESSAGE(STATUS (${CMAKE_SYSTEM} " system: using Windows native threads]")
#endif (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))

ADD_DEFINITIONS(-DUSE_CMAKE_CONFIG)

find_package(Threads)
if (NOT(CMAKE_HAVE_PTHREAD_H))
  if (CMAKE_SYSTEM_NAME MATCHES "Windows")
     ADD_DEFINITIONS(-D_FF_SYSTEM_HAVE_WIN_PTHREAD)
     message(STATUS "Using FF Pthread minport")
     set(FFHEADERS_PLAT
      ${FF}/platforms/pthread_minport_windows.h
      ${FF}/platforms/stdint.h
    )
  if (${CMAKE_SYSTEM} MATCHES Windows-5.1) 
    ADD_DEFINITIONS(-D_FF_WIN_XP)
    MESSAGE(STATUS ${CMAKE_SYSTEM} " system: using CondVar emulation")
  endif (${CMAKE_SYSTEM} MATCHES Windows-5.1) 
  else (CMAKE_SYSTEM_NAME MATCHES "Windows")
     message(FATAL_ERROR "Cannot find Pthreads")
  endif  (CMAKE_SYSTEM_NAME MATCHES "Windows")
endif (NOT(CMAKE_HAVE_PTHREAD_H))

if (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))
  if(CMAKE_HAVE_PTHREAD_H)
    MESSAGE(STATUS "Linking to pthreads")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
  endif(CMAKE_HAVE_PTHREAD_H)
endif  (NOT(CMAKE_SYSTEM_NAME MATCHES "Windows"))

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
# Avoid security in run time support and generate inlined intrinsics
  ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS) 
  if (CMAKE_BUILD_TYPE MATCHES Release)
    ADD_DEFINITIONS(-Oy -Oi -Ot -Ob2)
  endif (CMAKE_BUILD_TYPE MATCHES Release)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")


set( FF ${PROJECT_SOURCE_DIR}/ff )

## Experimental autoconf style configuration
INCLUDE (cmake.modules/ffconfig.cmake)


CHECK_INCLUDE_FILE("stdint.h"  HAVE_STDINT_H)

if (NOT(HAVE_STDINT_H)) 
  INCLUDE_DIRECTORIES(${FF}/platforms/)
endif (NOT(HAVE_STDINT_H))

#find_path(STDINT_INCLUDE_DIR stdint.h
#	HINTS /usr/include CMAKE_INCLUDE_PATH CMAKE_FRAMEWORK_PATH
#    )

#if (STDINT_INCLUDE_DIR MATCHES STDINT_INCLUDE_DIR-NOTFOUND)
#   message(STATUS  "System hasn't stdint.h, using FF" )
#   INCLUDE_DIRECTORIES(${FF}/platforms/stdint.h)
#   # On WinXP using our own  ${FF}/platforms/stdint.h
#endif (STDINT_INCLUDE_DIR MATCHES STDINT_INCLUDE_DIR-NOTFOUND)

set(FFHEADERS
 ${FF}/allocator.hpp
 ${FF}/barrier.hpp
 ${FF}/buffer.hpp
 ${FF}/config.hpp
 ${FF}/cycle.h
 ${FF}/dc.hpp
 ${FF}/dinout.hpp
 ${FF}/dnode.hpp
 ${FF}/dynlinkedlist.hpp
 ${FF}/dynqueue.hpp
 ${FF}/farm.hpp
 ${FF}/ff_queue.hpp
 ${FF}/fftree.hpp
 ${FF}/gsearch.hpp
 ${FF}/gt.hpp
 ${FF}/icl_hash.h
 ${FF}/lb.hpp
 ${FF}/make_unique.hpp
 ${FF}/map.hpp
 ${FF}/mapCUDAManaged.hpp
 ${FF}/mapper.hpp
 ${FF}/mapping_utils.hpp
 ${FF}/mdf.hpp
 ${FF}/multinode.hpp
 ${FF}/node.hpp
 ${FF}/oclallocator.hpp
 ${FF}/oclnode.hpp
 ${FF}/parallel_for.hpp
 ${FF}/parallel_for_internals.hpp
 ${FF}/pipeline.hpp
 ${FF}/poolEvolution.hpp
 ${FF}/poolEvolutionCUDA.hpp
 ${FF}/selector.hpp
 ${FF}/spin-lock.hpp
 ${FF}/squeue.hpp
 ${FF}/staticlinkedlist.hpp
 ${FF}/stencilReduce.hpp
 ${FF}/stencilReduceCUDA.hpp
 ${FF}/stencilReduceOCL.hpp
 ${FF}/stencilReduceOCL_macros.hpp
 ${FF}/svector.hpp
 ${FF}/sysdep.h
 ${FF}/task_internals.hpp
 ${FF}/taskf.hpp
 ${FF}/tpcallocator.hpp
 ${FF}/tpcnode.hpp
 ${FF}/ubuffer.hpp
 ${FF}/utils.hpp
 ${FF}/version.h
)

set(FFHEADERS_PLAT 
 ${FF}/platforms/getopt.h
 ${FF}/platforms/getopt.hpp
 ${FF}/platforms/liblfds.h
 ${FF}/platforms/platform.h
)


set(FFHEADERS_D
    ${FF}/d/inter.hpp
	${FF}/d/zmqImpl.hpp
	${FF}/d/zmqTransport.hpp
  )

if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
message(WARNING "Pinning on Power8 currently is not default" )
ADD_DEFINITIONS(-DNO_DEFAULT_MAPPING)
endif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")

# Examples currently not tested on arm
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
   find_library(ARM_RT_LIBRARY rt HINT /usr/lib DOC "arm rt library")
   if (ARM_RT_LIBRARY)
     link_libraries(${ARM_RT_LIBRARY})
   else (ARM_RT_LIBRARY) 
     message(FATAL_ERROR "Cannot find RT library that is required on arm processor ${ERR_MSG}")
   endif (ARM_RT_LIBRARY) 
    option( BUILD_EXAMPLES      "Build examples"    ON )
#   MESSAGE(WARNING "NOT Configuring >>examples<< directory (arm processor detected)")
else (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
   if (CMAKE_SYSTEM_NAME MATCHES "Windows")
     option( BUILD_EXAMPLES      "Build examples"        ON )
     MESSAGE(STATUS "Configuring >>examples<< directory")
   else (CMAKE_SYSTEM_NAME MATCHES "Windows")
     option( BUILD_EXAMPLES      "Build examples"        ON )
     MESSAGE(STATUS "Configuring >>examples<< directory")
   endif (CMAKE_SYSTEM_NAME MATCHES "Windows")
endif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")

#       NAME                "Description"           DEFAULT
option( BUILD_TESTS         "Build tests"           ON ) 
MESSAGE(STATUS "Configuring >>tests<< directory")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY )

add_custom_target( uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" )

install( FILES ${FFHEADERS} DESTINATION include/ff/ )
install( FILES ${FFHEADERS_PLAT} DESTINATION include/ff/platforms )
install( FILES ${FFHEADERS_D} DESTINATION include/ff/d )

if( BUILD_EXAMPLES )
#    enable_testing()
#    add_subdirectory( examples )
endif( BUILD_EXAMPLES )

if( BUILD_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif( BUILD_TESTS )

#FIND_PACKAGE( zeromq )

