# FIXME some tests not compiling

set(TESTS
# funny-pipeline
l1_ff_nodes_graph
l2_cyclic_graph
l2_cyclic_graph_trace
l2_generic_dag
l2_generic_dag_pinning
# l2_mapping_utils
)

foreach( t ${TESTS} )
  add_executable( ${t} ${t}.cpp)
  target_include_directories(${t} PRIVATE
                             $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>)
  target_link_libraries( ${t} ${CMAKE_THREAD_LIBS_INIT} )
endforeach( t )

set_target_properties (l2_cyclic_graph_trace PROPERTIES COMPILE_DEFINITIONS "TRACE_FASTFLOW ")
# set_target_properties (l2_mapping_utils PROPERTIES COMPILE_DEFINITIONS "TRACE_FASTFLOW ")

add_test( l1_ff_nodes_graph ${CMAKE_CURRENT_BINARY_DIR}/l1_ff_nodes_graph)
add_test(  l2_generic_dag ${CMAKE_CURRENT_BINARY_DIR}/l2_generic_dag )
add_test( l2_cyclic_graph ${CMAKE_CURRENT_BINARY_DIR}/l2_cyclic_graph 10 10)
add_test( l2_cyclic_graph_trace ${CMAKE_CURRENT_BINARY_DIR}/l2_cyclic_graph 10 10)
# add_test( l2_mapping_utils ${CMAKE_CURRENT_BINARY_DIR}/l2_mapping_utils 10 10)
add_test( l2_generic_dag_pinning ${CMAKE_CURRENT_BINARY_DIR}/l2_generic_dag_pinning)

foreach( t ${TESTS} )
  set_tests_properties ( ${t} PROPERTIES TIMEOUT 180)
endforeach ()