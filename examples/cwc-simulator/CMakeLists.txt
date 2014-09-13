# Author: Marco Aldinucci
# Date: Oct 9, 2009

#cmake_minimum_required(VERSION 2.8)

add_definitions(${Boost_LIB_DIAGNOSTIC_DEFINITIONS})

message(STATUS "Boost Libraries dir: " ${Boost_LIBRARY_DIRS})
message(STATUS "Boost Libraries: " ${Boost_LIBRARIES})

set (PARSER_SRCS_DIR parser)
set ( CWC_FF_SRCS driver.cpp Compartment.cpp
  Monitor.cpp Simulation.cpp
  ${PARSER_SRCS_DIR}/parser.cpp
  ${PARSER_SRCS_DIR}/Driver.cpp
  OCompartment.cpp Species.cpp
  ${PARSER_SRCS_DIR}/scanner.cpp PCompartment.cpp
  SubMatch.cpp Match.cpp
  PSpecies.cpp SubMatchSet.cpp
  MatchSet.cpp Model.cpp Rule.cpp
  filter.cpp kmeans_seq.cpp qt.cpp
  output.cpp gnuplot.cpp utils.cpp )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/parser
    ${Boost_INCLUDE_DIRS}
    )
#LINK_DIRECTORIES(${Boost_LIBRARY_DIRS})

if(COMMAND cmake_policy)
  cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

find_package(ZeroMQ)
if(NOT ZeroMQ_FOUND)
  message (WARNING " 0mq not found: cannot configure distributed mode")
endif()

#available flags: #-DLOG -DTRACE_FASTFLOW -DDATASIZE -DPINNING -DTIME -DGRAIN

#default: (stochastic) shared-memory, par. simulations, par. statistics
add_executable ( scwc ${CWC_FF_SRCS} )

target_link_libraries (scwc   ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT})
set_target_properties (scwc PROPERTIES COMPILE_FLAGS "-DFF_ALLOCATOR -DUSE_FF_ACCEL -DUSE_STAT_ACCEL -DBARRIER_T=Barrier")

#distributed #    ${FFHEADERS_D}
if(ZeroMQ_FOUND)
  include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ZeroMQ_INCLUDE_DIRS}
    )
 add_executable ( scwc_distr ${CWC_FF_SRCS} )
 #link_directories( scwc_distr ${Boost_LIBRARY_DIRS})
  target_link_libraries (scwc_distr  ${Boost_LIBRARIES}  ${CMAKE_THREAD_LIBS_INIT} ${ZeroMQ_LIBRARIES})
  set_target_properties (scwc_distr PROPERTIES COMPILE_FLAGS "-DUSE_FF_ACCEL -DUSE_FF_DISTR -DFF_ALLOCATOR -DUSE_STAT_ACCEL -DBARRIER_T=Barrier")
endif()

add_test( scwc ${CMAKE_CURRENT_BINARY_DIR}/scwc -i ${CMAKE_CURRENT_SOURCE_DIR}/examples/ecoli/low.cwc -o ecoli -t 400000 -s 2400 -n 10 -w 2 )
