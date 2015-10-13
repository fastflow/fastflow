include_directories(
    ${PROJECT_SOURCE_DIR}
    )




#file(MAKE_DIRECTORY "UNBOUNDED_MPMC")
#file(MAKE_DIRECTORY "BOUNDED_MPMC")


#add_executable( test_mpmc test_mpmc.cpp )
#add_executable( test_bmpmc test_bmpmc.cpp )

#set_target_properties (test_mpmc PROPERTIES COMPILE_DEFINITIONS "DEFINE-HERE")


set( TESTS test_mpmc test_bmpmc)

foreach( t ${TESTS} )
	 add_executable(${t} ${t}.cpp)
	 add_executable(${t}_UNBOUND ${t}.cpp)
	 set_target_properties (${t}_UNBOUND PROPERTIES COMPILE_DEFINITIONS "UNBOUNDED_MPMC")  
endforeach( t )



foreach( t ${TESTS} )
    target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} )
    target_link_libraries( ${t}_UNBOUND ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

foreach( t ${TESTS} )
    add_test( ${t} ${CMAKE_CURRENT_BINARY_DIR}/${t} )
    add_test( ${t}_UNBOUND ${CMAKE_CURRENT_BINARY_DIR}/${t}_UNBOUND )
endforeach( t )
