project( mc-fastflow )

cmake_minimum_required( VERSION 2.4 )

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
   ) 

set(FFHEADERS_AT
    ${FF}/atomic/atomic.h
    ${FF}/atomic/atomic-i386.h
    ${FF}/atomic/atomic-ppc.h
    ${FF}/atomic/atomic-x86_64.h
   )

#       NAME                "Description"           DEFAULT
option( BUILD_EXAMPLES      "Build examples"        ON )
option( BUILD_TESTS         "Build tests"           ON ) 

configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY )

add_custom_target( uninstall
    "${CMAKE_COMMAND}" -P "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake" )

install( FILES ${FFHEADERS} DESTINATION /include/ff/ )
install( FILES ${FFHEADERS_AT} DESTINATION /include/ff/atomic )

if( BUILD_EXAMPLES )
    add_subdirectory( examples )
endif( BUILD_EXAMPLES )

if( BUILD_TESTS )
    enable_testing()
    add_subdirectory( tests )
endif( BUILD_TESTS )
