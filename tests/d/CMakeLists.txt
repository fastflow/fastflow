include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ZeroMQ_INCLUDE_DIRS}
    )

include_directories( ${ZMQ_INCLUDE_DIRS} )

#link_directories(${ZMQ_LIBRARIES})

add_executable( allgather allgather.cpp )  
add_executable( broadcast broadcast.cpp ) 
add_executable( dmap dmap.cpp ) 
add_executable( fromany fromany.cpp )    
add_executable( farm farm.cpp ) 
    

set( DTESTS allgather broadcast dmap fromany farm)

foreach( t ${DTESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT}  ${ZMQ_LIBRARIES})
endforeach( t )

#add_test( allgather01 ${CMAKE_CURRENT_BINARY_DIR}/allgather ciccio 1 0 localhost:10002 & )
#add_test( allgather11 ${CMAKE_CURRENT_BINARY_DIR}/allgather ciccio 0 1 localhost:10002 )
