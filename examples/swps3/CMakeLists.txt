#
# Copyright (c) 2007-2008 ETH Zürich, Institute of Computational Science
#
# Permission is hereby granted, free of charge, to any person
# obtaining a copy of this software and associated documentation
# files (the "Software"), to deal in the Software without
# restriction, including without limitation the rights to use,
# copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the
# Software is furnished to do so, subject to the following
# conditions:
#
# The above copyright notice and this permission notice shall be
# included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
# OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
# NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
# HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
# WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
# OTHER DEALINGS IN THE SOFTWARE.

CMAKE_MINIMUM_REQUIRED(VERSION 2.4)
PROJECT(swps3)

SET(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/CMakeScripts ${PROJECT_BINARY_DIR}/CMakeScripts)

################################################################################
#  _            _       
# | |_ ___  ___| |_ ___ 
# | __/ _ \/ __| __/ __|
# | ||  __/\__ \ |_\__ \
#  \__\___||___/\__|___/
#                       
################################################################################

# test for platform(?)
# test for sysconf(NPROCESSORS)
# test for fork (?)

################################################################################
#                                              _ _           
#  ___ _ __  _   _    ___ ___  _ __ ___  _ __ (_) | ___ _ __ 
# / __| '_ \| | | |  / __/ _ \| '_ ` _ \| '_ \| | |/ _ \ '__|
# \__ \ |_) | |_| | | (_| (_) | | | | | | |_) | | |  __/ |   
# |___/ .__/ \__,_|  \___\___/|_| |_| |_| .__/|_|_|\___|_|   
#     |_|                               |_|                  
################################################################################

FIND_PROGRAM(SPU_C_COMPILER $ENV{SPECC} spu-elf-gcc spu-xlc DOC "C compiler for the Cell/BE SPU.")
FIND_PROGRAM(SPU_CXX_COMPILER $ENV{SPECXX} spu-elf-c++ spu-elf-g++ spu-xlC DOC "C++ compiler for the Cell/BE SPU.")
FIND_PROGRAM(PATH_EMBEDSPU embedspu DOC "Tool for embedding a SPU binary into a PPU library.")
IF(SPU_C_COMPILER AND SPU_CXX_COMPILER AND PATH_EMBEDSPU)
   SET(HAVE_SPU_TOOLS ON)
ELSE(SPU_C_COMPILER AND SPU_CXX_COMPILER AND PATH_EMBEDSPU)
   SET(HAVE_SPU_TOOLS OFF)
ENDIF(SPU_C_COMPILER AND SPU_CXX_COMPILER AND PATH_EMBEDSPU)

SET(SPU_C_FLAGS "-O3 -funroll-all-loops -msmall-mem -mbranch-hints" CACHE STRING "Flags used by the compiler during all build types.")
SET(SPU_CXX_FLAGS "-O3 -funroll-all-loops -msmall-mem -mbranch-hints" CACHE STRING "Flags used by the compiler during all build types.")

################################################################################
#                 _                        _   _                 
#   ___ __ _  ___| |__   ___    ___  _ __ | |_(_) ___  _ __  ___ 
#  / __/ _` |/ __| '_ \ / _ \  / _ \| '_ \| __| |/ _ \| '_ \/ __|
# | (_| (_| | (__| | | |  __/ | (_) | |_) | |_| | (_) | | | \__ \
#  \___\__,_|\___|_| |_|\___|  \___/| .__/ \__|_|\___/|_| |_|___/
#                                   |_|                          
################################################################################

SET(WITH_DEBUG OFF CACHE BOOL "enable debug output")
SET(WITH_STATIC_LIBSWPS3 OFF CACHE BOOL "build and link the libswps3 statically")
SET(WITH_SSE ON CACHE BOOL "enable code using SSE2 instructions")
SET(WITH_CELL_SPU ${HAVE_SPU_TOOLS} CACHE BOOL "enable code generation for the Cell/BE SPU")
SET(WITH_ALTIVEC OFF CACHE BOOL "enable code using Altivec instructions")
SET(WITH_UNROLL_LOOPS ON CACHE BOOL "unroll inner loops once (recommended)")
MARK_AS_ADVANCED(WITH_UNROLL_LOOPS)

################################################################################
#                                       ___     _                       _       
#  ___  ___  _   _ _ __ ___ ___  ___   ( _ )   | |_ __ _ _ __ __ _  ___| |_ ___ 
# / __|/ _ \| | | | '__/ __/ _ \/ __|  / _ \/\ | __/ _` | '__/ _` |/ _ \ __/ __|
# \__ \ (_) | |_| | | | (_|  __/\__ \ | (_>  < | || (_| | | | (_| |  __/ |_\__ \
# |___/\___/ \__,_|_|  \___\___||___/  \___/\/  \__\__,_|_|  \__, |\___|\__|___/
#                                                            |___/              
################################################################################

SET(LIBRARY_SOURCES fasta.c matrix.c DynProgr_scalar.c debug.c)
SET(SPU_BACKEND_SOURCES_C DynProgr_SPE_main.c)
SET(SPU_BACKEND_SOURCES_CXX DynProgr_SPE_functions.cc)
SET(SPU_SOURCES DynProgr_PPU.c DynProgr_SPE.o)
SET(ALTIVEC_SOURCES DynProgr_altivec.cc)
SET(SSE_SOURCES DynProgr_sse_byte.c DynProgr_sse_short)

ADD_EXECUTABLE(swps3 swps3.c)
TARGET_LINK_LIBRARIES(swps3 libswps3)

IF(WITH_SSE)
   SET(LIBRARY_SOURCES ${LIBRARY_SOURCES} ${SSE_SOURCES})
ENDIF(WITH_SSE)
IF(WITH_ALTIVEC)
   SET(LIBRARY_SOURCES ${LIBRARY_SOURCES} ${ALTIVEC_SOURCES})
ENDIF(WITH_ALTIVEC)
IF(WITH_CELL_SPU)
   SET(LIBRARY_SOURCES ${LIBRARY_SOURCES} ${SPU_SOURCES})
ENDIF(WITH_CELL_SPU)

IF(WITH_STATIC_LIBSWPS3)
ADD_LIBRARY(libswps3 STATIC ${LIBRARY_SOURCES})
ELSE(WITH_STATIC_LIBSWPS3)
ADD_LIBRARY(libswps3 SHARED ${LIBRARY_SOURCES})
ENDIF(WITH_STATIC_LIBSWPS3)
SET_TARGET_PROPERTIES(libswps3 PROPERTIES OUTPUT_NAME swps3)

IF(WITH_CELL_SPU)
   SET(SPU_HEADERS DynProgr_SPE_functions.h DynProgr_SPE.h DynProgr_PPU.h swps3.h)

   FOREACH(SOURCE ${SPU_BACKEND_SOURCES_C})
      STRING(REGEX REPLACE "\\.c$" ".oo" TARGET ${SOURCE})
      ADD_CUSTOM_COMMAND(OUTPUT ${TARGET}
	 COMMAND ${SPU_C_COMPILER} ARGS ${SPU_C_FLAGS} ${SOURCE} -o ${TARGET}
	 DEPENDS ${SOURCE} ${SPU_HEADERS})
   ENDFOREACH(SOURCE)

   FOREACH(SOURCE ${SPU_BACKEND_SOURCES_CXX})
      STRING(REGEX REPLACE "\\.cc$" ".oo" TARGET ${SOURCE})
      ADD_CUSTOM_COMMAND(OUTPUT ${TARGET}
	 COMMAND ${SPU_CXX_COMPILER} ARGS ${SPU_CXX_FLAGS} ${SOURCE} -o ${TARGET}
	 DEPENDS ${SOURCE} ${SPU_HEADERS})
   ENDFOREACH(SOURCE)

   #   STRING(REGEX REPLACE "\\.cc$|\\.c$" ".oo" SPU_OBJECTS ${SPU_BACKEND_SOURCES_C} ${SPU_BACKEND_SOURCES_CXX})
   SET(SPU_OBJECTS DynProgr_SPE_functions.oo DynProgr_SPE_main.oo)
   ADD_CUSTOM_COMMAND(OUTPUT DynProgr_SPE
	   COMMAND ${SPU_CXX_COMPILER} ARGS ${SPU_CXX_FLAGS} ${SPU_OBJECTS} -o DynProgr_SPE
	   DEPENDS ${SPU_OBJECTS})
   ADD_CUSTOM_COMMAND(OUTPUT DynProgr_SPE.o
	   COMMAND ${PATH_EMBEDSPU} ARGS spe_dynprogr_handle DynProgr_SPE DynProgr_SPE.o
	   DEPENDS DynProgr_SPE)
ENDIF(WITH_CELL_SPU)

################################################################################
#                                                              
#  _ __  _ __ ___ _ __  _ __ ___   ___ ___  ___ ___  ___  _ __ 
# | '_ \| '__/ _ \ '_ \| '__/ _ \ / __/ _ \/ __/ __|/ _ \| '__|
# | |_) | | |  __/ |_) | | | (_) | (_|  __/\__ \__ \ (_) | |   
# | .__/|_|  \___| .__/|_|  \___/ \___\___||___/___/\___/|_|   
# |_|            |_|                                           
#      _       __ _       _ _   _                 
#   __| | ___ / _(_)_ __ (_) |_(_) ___  _ __  ___ 
#  / _` |/ _ \ |_| | '_ \| | __| |/ _ \| '_ \/ __|
# | (_| |  __/  _| | | | | | |_| | (_) | | | \__ \
#  \__,_|\___|_| |_|_| |_|_|\__|_|\___/|_| |_|___/
#                                                 
################################################################################

#CONFIGURE_FILE(config.h.in config.h)
IF(WITH_UNROLL_LOOPS)
   ADD_DEFINITIONS(-DUNROLL)
ENDIF(WITH_UNROLL_LOOPS)
   
IF(WITH_DEBUG)
   ADD_DEFINITIONS(-DDEBUG)
ENDIF(WITH_DEBUG)

IF(WITH_SSE)
   #TODO icc does only recognize -msse3
   ADD_DEFINITIONS(-msse2)
ENDIF(WITH_SSE)

IF(WITH_ALTIVEC)
   ADD_DEFINITIONS(-maltivec)
ENDIF(WITH_ALTIVEC)

IF(WITH_CELL_SPU)
   ADD_DEFINITIONS(-D__PS3)
ENDIF(WITH_CELL_SPU)

################################################################################
#  _           _        _ _       _   _             
# (_)_ __  ___| |_ __ _| | | __ _| |_(_) ___  _ __  
# | | '_ \/ __| __/ _` | | |/ _` | __| |/ _ \| '_ \ 
# | | | | \__ \ || (_| | | | (_| | |_| | (_) | | | |
# |_|_| |_|___/\__\__,_|_|_|\__,_|\__|_|\___/|_| |_|
#                                                   
################################################################################

INSTALL(TARGETS swps3 DESTINATION bin)
INSTALL(TARGETS libswps3 DESTINATION lib)
SET(INSTALLABLE_HEADERS DynProgr_PPU.h DynProgr_sse_byte.h DynProgr_sse_short.h DynProgr_altivec.h DynProgr_scalar.h swps3.h)
INSTALL(FILES ${INSTALLABLE_HEADERS} DESTINATION include/swps3)

