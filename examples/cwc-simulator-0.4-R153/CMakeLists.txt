# Author: Marco Aldinucci
# Date: Oct 9, 2009

#cmake_minimum_required(VERSION 2.6)

add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

set ( CWC_FF_SRCS scwc.cpp  Compartment.cpp  Monitor.cpp
Simulation.cpp   parser.cpp Driver.cpp       OCompartment.cpp
Species.cpp      scanner.cpp Gillespie.cpp    PCompartment.cpp
SubMatch.cpp   Match.cpp        PSpecies.cpp      SubMatchSet.cpp
statistics.cpp MatchSet.cpp     ProgressBars.cpp   Model.cpp
Rule.cpp    output.cpp kmeans_seq.cpp )


set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}")


#set (HYBRID "")
find_package(GSL)
if (GSL_FOUND)
MESSAGE(STATUS   "GSL found, configuring CWC simulator for hybrid mode")
set ( CWC_FF_SRCS_HYBRID ${CWC_FF_SRCS} ode.cpp) 
#add_definitions(-DHYBRID)
include_directories ( ${GSL_INCLUDE_DIRS} )
endif (GSL_FOUND)

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CMAKE_CURRENT_SOURCE_DIR}
	${CMAKE_CURRENT_SOURCE_DIR}/include		
    ${Boost_INCLUDE_DIRS}
    )

if(COMMAND cmake_policy)
		   cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_executable ( scwc ${CWC_FF_SRCS} )
if (GSL_FOUND)
add_executable ( scwc_hybrid ${CWC_FF_SRCS_HYBRID} )
endif (GSL_FOUND)

#set(Boost_USE_STATIC_LIBS   ON)
#set(Boost_USE_MULTITHREADED ON)
#set(BOOST_LIBS thread)
#link_directories ( ${Boost_LIBRARY_DIRS}  )
#target_link_libraries (scwc  ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries (scwc  ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} )
#target_link_libraries (scwc  ${CMAKE_THREAD_LIBS_INIT} )

set_target_properties ( scwc   PROPERTIES COMPILE_FLAGS "-DUSE_FF_ACCEL -DFF_ALLOCATOR  -DLOCKFREE")

if (GSL_FOUND)
   target_link_libraries (scwc_hybrid  ${CMAKE_THREAD_LIBS_INIT} ${Boost_LIBRARIES} ${GSL_LIBRARIES} ${GSLCBLAS_LIBRARIES})
   set_target_properties ( scwc_hybrid   PROPERTIES COMPILE_FLAGS "-O3 -Wall -DUSE_FF_ACCEL -DFF_ALLOCATOR  -DLOCKFREE -DHYBRID")
endif  (GSL_FOUND)

add_test( scwc   ${CMAKE_CURRENT_BINARY_DIR}/scwc -i ${CMAKE_CURRENT_SOURCE_DIR}/examples/ecoli/low.cwc -o ecoli -t 600000 -s 2400 -n 10 -w 6 )
