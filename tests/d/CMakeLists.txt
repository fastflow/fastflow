include_directories(
    ${PROJECT_SOURCE_DIR}
    ${ZeroMQ_INCLUDE_DIRS}
    )

include_directories( ${ZMQ_INCLUDE_DIRS} )

#link_directories(${ZMQ_LIBRARIES})

    
set( DTESTS
allgather  dmap2      lat11      pipe_farm2    test_dinout
broadcast  farm       lat11_v2   scatter       test_gw
bw11       farm_farm  ondemand   test11_pipe   test_marshal
dmap       fromany    pipe_farm  test11_torus  unicast
)

foreach( t ${DTESTS} )
	 add_executable( ${t} ${t}.cpp )
    	 target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT}  ${ZMQ_LIBRARIES})
endforeach( t )

#add_test( allgather01 ${CMAKE_CURRENT_BINARY_DIR}/allgather ciccio 1 0 localhost:10002 & )
#add_test( allgather11 ${CMAKE_CURRENT_BINARY_DIR}/allgather ciccio 0 1 localhost:10002 )
