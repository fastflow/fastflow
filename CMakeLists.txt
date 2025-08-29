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
    MESSAGE( STATUS "Build Type from command line " ${CMAKE_BUILD_TYPE})
ELSE()
    SET(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build,
options are: None(CMAKE_CXX_FLAGS or CMAKE_C_FLAGS used) Debug Release
RelWithDebInfo MinSizeRel.")
    MESSAGE( STATUS "Default Build Type: Release  (change with
      -DCMAKE_BUILD_TYPE=Debug | Release | RelWithDebInfo | MinSizeRel")
ENDIF()

cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0054 NEW)

project( mc-fastflow )

if (CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
	SET (PROJECT_SOURCE_DIR /cygdrive/c/Users/Luca\ Ferrucci/git/fastflow)
	SET (CMAKE_REQUIRED_INCLUDES /cygdrive/c/cygwin64/usr/include/proc)
endif (CMAKE_SYSTEM_NAME MATCHES "CYGWIN")

SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules )
FILE (COPY ${PROJECT_SOURCE_DIR}/cmake.modules/templates/ffconfig.h DESTINATION ${PROJECT_SOURCE_DIR}/cmake.modules/)

INCLUDE (${CMAKE_ROOT}/Modules/CheckIncludeFile.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckTypeSize.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckFunctionExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckCXXSourceCompiles.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/TestBigEndian.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckSymbolExists.cmake)
INCLUDE (${CMAKE_ROOT}/Modules/CheckLibraryExists.cmake)

MESSAGE( STATUS "Detected processor is " ${CMAKE_SYSTEM_PROCESSOR})
MESSAGE( STATUS "CMAKE_C_COMPILER: " ${CMAKE_C_COMPILER} )
MESSAGE( STATUS "CMAKE_CXX_COMPILER: " ${CMAKE_CXX_COMPILER} )

set( FF ${PROJECT_SOURCE_DIR}/ff )

if (${CMAKE_CXX_COMPILER_ID} STREQUAL "Clang")
    MESSAGE( STATUS "C++ Compiler is Clang")
    SET(FFCM_HAS_CLANGXX "TRUE")
    SET(CMAKE_CXX_FLAGS "-std=c++20")
    SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-deprecated-register")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
    MESSAGE( STATUS "C++ Compiler is GNU")
    if (CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
    	ADD_DEFINITIONS(-D_POSIX_C_SOURCE=200112L)
    endif(CMAKE_SYSTEM_NAME MATCHES "CYGWIN")
    execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GXX_VERSION)
    if (NOT(GXX_VERSION VERSION_GREATER 10.0))
        set(GCC_TOO_OLD 1)
        MESSAGE( WARNING "g++ Version < 10.0, some targets will be skipped")
        MESSAGE( STATUS "Use CC/CXX env variable to define a newer compiler, e.g. export CC=/usr/bin/gcc-10; export CXX=/usr/bin/g++-10 ")
    else (NOT(GXX_VERSION VERSION_GREATER 10.0))
        ADD_DEFINITIONS(-std=c++20) # or -std=c++0x
    endif (NOT(GXX_VERSION VERSION_GREATER 10.0))
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "Intel")
    MESSAGE( STATUS "C++ Compiler is Intel")
    MESSAGE( WARNING "Compliance check not fully implemented yet")
elseif (${CMAKE_CXX_COMPILER_ID} STREQUAL "MSVC")
    MESSAGE( STATUS "C++ Compiler is Visual Studio")
endif()

ADD_DEFINITIONS(-DUSE_CMAKE_CONFIG)

if (CMAKE_SYSTEM_NAME MATCHES "Windows")
    ADD_DEFINITIONS(-DNO_DEFAULT_MAPPING)
    MESSAGE( WARNING "Pinning on Win platforn currently is not default" )
    if(CMAKE_CL_64)
        MESSAGE( STATUS "64-bit microsoft compiler found")
        ADD_DEFINITIONS(-D_WIN64)
        ADD_DEFINITIONS(-D_AMD64_)
        ADD_DEFINITIONS(-DNOMINMAX)
    else(CMAKE_CL_64)
        MESSAGE( STATUS "32-bit microsoft compiler found")
    endif(CMAKE_CL_64)
    ADD_DEFINITIONS(-D_FF_SYSTEM_HAVE_WIN_PTHREAD)
    MESSAGE( STATUS "Using pthreads4w library")
    set(PTHREAD_INCLUDE_DIR ${FF}/platforms/winlibs/include)
    set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${FF}/platforms/winlibs/pthreadVC3.lib)
    if (${CMAKE_SYSTEM} MATCHES Windows-5.1)
        ADD_DEFINITIONS(-D_FF_WIN_XP)
        MESSAGE( STATUS ${CMAKE_SYSTEM} " system: using CondVar emulation")
    endif (${CMAKE_SYSTEM} MATCHES Windows-5.1)
    # Avoid security in run time support and generate inlined intrinsics
    ADD_DEFINITIONS(-D_CRT_SECURE_NO_WARNINGS)
    if (CMAKE_BUILD_TYPE MATCHES Release)
        ADD_DEFINITIONS(/O2 /Ob2)
    endif (CMAKE_BUILD_TYPE MATCHES Release)
else (CMAKE_SYSTEM_NAME MATCHES "Windows")
        find_package(Threads REQUIRED)
        if(Threads_FOUND)
            MESSAGE( STATUS "Linking to pthreads")
            set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
            SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -lpthread")
        else(Threads_FOUND)
            MESSAGE(FATAL_ERROR "Cannot find Pthreads")
        endif (Threads_FOUND)
        if (CMAKE_BUILD_TYPE MATCHES Release)
            if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
                ADD_DEFINITIONS(-Wall -Wextra -O3 -march=native )
            endif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        elseif(CMAKE_BUILD_TYPE MATCHES Debug)
            if (${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
                ADD_DEFINITIONS(-Wall -Wextra -g)
            endif(${CMAKE_CXX_COMPILER_ID} STREQUAL "GNU")
        endif(CMAKE_BUILD_TYPE MATCHES Release)
endif (CMAKE_SYSTEM_NAME MATCHES "Windows")

## Experimental autoconf style configuration
INCLUDE (cmake.modules/ffconfig.cmake)

CHECK_INCLUDE_FILE("stdint.h"  HAVE_STDINT_H)

if (NOT(HAVE_STDINT_H))
    INCLUDE_DIRECTORIES(${FF}/platforms/)
endif (NOT(HAVE_STDINT_H))

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
    ${FF}/faas_ff.hpp
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
    ${FF}/utimer.hpp
    ${FF}/version.h)

set(FFHEADERS_PLAT
    ${FF}/platforms/getopt.h
    ${FF}/platforms/getopt.hpp
    ${FF}/platforms/liblfds.h
    ${FF}/platforms/platform.h)

set(FFHEADERS_D
    ${FF}/d/inter.hpp
    ${FF}/d/zmqImpl.hpp
    ${FF}/d/zmqTransport.hpp)

if (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")
    MESSAGE( WARNING "Pinning on Power8 currently is not default" )
    ADD_DEFINITIONS(-DNO_DEFAULT_MAPPING)
endif (CMAKE_SYSTEM_PROCESSOR MATCHES "ppc")

# Examples currently not tested on arm
if (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    find_library(ARM_RT_LIBRARY rt HINT /usr/lib DOC "arm rt library")
    if (ARM_RT_LIBRARY)
        link_libraries(${ARM_RT_LIBRARY})
    else (ARM_RT_LIBRARY)
        MESSAGE( FATAL_ERROR "Cannot find RT library that is required on arm processor ${ERR_MSG}")
    endif (ARM_RT_LIBRARY)
    option( BUILD_EXAMPLES      "Build examples"    ON )
else (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")
    if (CMAKE_SYSTEM_NAME MATCHES "Windows")
        option( BUILD_EXAMPLES      "Build examples"        ON )
        MESSAGE( STATUS "Configuring >>examples<< directory")
    else (CMAKE_SYSTEM_NAME MATCHES "Windows")
        option( BUILD_EXAMPLES      "Build examples"        ON )
        MESSAGE( STATUS "Configuring >>examples<< directory")
    endif (CMAKE_SYSTEM_NAME MATCHES "Windows")
endif (CMAKE_SYSTEM_PROCESSOR MATCHES "arm")

#       NAME                "Description"           DEFAULT
option( BUILD_TESTS         "Build tests"           ON )
MESSAGE( STATUS "Configuring >>tests<< directory")
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY )

add_custom_target( uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" )

install( FILES ${FFHEADERS} DESTINATION include/ff/ )
install( FILES ${FFHEADERS_PLAT} DESTINATION include/ff/platforms )
install( FILES ${FFHEADERS_D} DESTINATION include/ff/d )

if( BUILD_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif( BUILD_TESTS )