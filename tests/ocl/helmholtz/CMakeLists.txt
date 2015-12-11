# Author: MarcoA

include_directories(
    ${PROJECT_SOURCE_DIR}
    )

ADD_DEFINITIONS(-DFF_OPENCL ) #-DVERIFY

include_directories( ${OPENCL_INCLUDE_DIRS} )


set( TESTS                               
helmholtz
)


foreach( t ${TESTS} )
    add_executable( ${t} ${t}.cpp)
endforeach( t )

foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} ${OPENCL_LIBRARIES} )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
endforeach( t )




add_custom_command(TARGET helmholtz POST_BUILD
                   COMMAND ${CMAKE_COMMAND} -E copy_if_different
                       ${CMAKE_CURRENT_SOURCE_DIR}/ff_opencl_datatypes.cl
                       $<TARGET_FILE_DIR:helmholtz>
                 )