set ( TESTS
    simplest
    test1 test1b test2 test3 test3b test3_farm test4 test5 test6 test7 test8
    perf_test1
    test_accelerator test_accelerator2 test_accelerator3
    test_accelerator_farm+pipe test_accelerator_pipe
    test_ofarm test_ofarm2
    test_accelerator_ofarm test_accelerator_ofarm_multiple_freezing
    test_accelerator_pipe+farm test_farm+pipe test_farm+pipe2
    test_FFConvFaaS test_faas_function test_faas
    test_freeze
    test_masterworker bench_masterworker
    test_multi_masterworker test_pipe+masterworker
    test_scheduling
    test_dt test_torus test_torus2
    perf_test_alloc1 perf_test_alloc2 perf_test_alloc3
    perf_test_noalloc test_uBuffer test_sendq test_spinBarrier
    test_multi_input test_multi_input2 test_multi_input3 test_multi_input4 test_multi_input5 test_multi_input6 test_multi_input7 test_multi_input8 test_multi_input9 test_multi_input10 test_multi_input11
    test_accelerator+pinning
    test_dataflow test_dataflow2
    test_noinput_pipe
    test_stopstartthreads test_stopstartthreads2 test_stopstartthreads3 test_stopstartall
    test_MISD
    test_parfor test_parfor2 test_parforpipereduce
    test_dotprod_parfor test_parfor_unbalanced test_parfor_multireduce test_parfor_multireduce2
    test_lb_affinity
    test_farm test_farm2
    test_pipe test_pipe2
    perf_parfor perf_parfor2
    test_graphsearch
    test_multi_output test_multi_output2 test_multi_output3 test_multi_output4 test_multi_output5 test_multi_output6
    test_pool1 test_pool2 test_pool3
    test_devicequery test_map test_mdf test_taskf latptr11 test_taskcallbacks test_eosw test_nodeselector test_stats test_dc
    test_combine test_combine1 test_combine2 test_combine3 test_combine4 test_combine5 test_combine6 test_combine7 test_combine8 test_combine9 test_combine10 test_combine11 test_combine12 test_combine13 test_combine14
    test_all-to-all test_all-to-all2 test_all-to-all3 test_all-to-all4 test_all-to-all5 test_all-to-all6 test_all-to-all7 test_all-to-all8 test_all-to-all9 test_all-to-all10 test_all-to-all11 test_all-to-all12 test_all-to-all13 test_all-to-all14 test_all-to-all15 test_all-to-all16
    test_optimize test_optimize2 test_optimize3 test_optimize4 test_optimize5
    test_all-or-none test_farm+farm test_farm+A2A test_farm+A2A2)

set ( TESTS
    test_faas_function test_faas test_faas_raw)

foreach( t ${TESTS} )
   
    set(TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/${t}.cpp")
   
    if(NOT EXISTS ${TEST_SRC})
        set(TEST_SRC "${CMAKE_CURRENT_SOURCE_DIR}/FaaS/${t}.cpp")
    endif()

    if(NOT EXISTS ${TEST_SRC})
        message(FATAL_ERROR "File sorgente per ${t} non trovato in nessuna directory")
    endif()

    if (t MATCHES "test_FFConvFaaS")
        find_package(CURL REQUIRED)
        if(CURL_FOUND)            
            set (CMAKE_REQUIRED_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES} ${CURL_LIBRARIES}) 
        else(CURL_FOUND)
            message(FATAL_ERROR "CURL library not found")
        endif(CURL_FOUND)
    endif(t MATCHES "test_FFConvFaaS")

    if (t MATCHES "test_faas")
        set(PROJECT_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/ff/FaaS/include)
        set(FAAS_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/ff/FaaS)
        add_definitions(-DDFF_EXCLUDE_MPI)
    endif(t MATCHES "test_faas")

    add_executable( ${t}_NONBLOCKING  ${TEST_SRC})
    target_include_directories(${t}_NONBLOCKING PRIVATE  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<BUILD_INTERFACE:${FAAS_INCLUDE_DIR}>  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}> $<BUILD_INTERFACE: ${CURL_INCLUDE_DIRS}> $<BUILD_INTERFACE:${PTHREAD_INCLUDE_DIR}> )
    target_link_libraries(${t}_NONBLOCKING PRIVATE ${CMAKE_REQUIRED_LIBRARIES})
    add_test( ${t}_NONBLOCKING ${CMAKE_CURRENT_BINARY_DIR}/${t}_NONBLOCKING )
    set_tests_properties ( ${t}_NONBLOCKING PROPERTIES TIMEOUT 180)

    add_executable( ${t}_BLOCKING ${TEST_SRC})
    target_include_directories(${t}_BLOCKING PRIVATE  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}> $<BUILD_INTERFACE:${FAAS_INCLUDE_DIR}>  $<BUILD_INTERFACE:${PROJECT_INCLUDE_DIR}> $<BUILD_INTERFACE:${CURL_INCLUDE_DIRS}> $<BUILD_INTERFACE:${PTHREAD_INCLUDE_DIR}> )
    target_link_libraries(${t}_BLOCKING PRIVATE ${CMAKE_REQUIRED_LIBRARIES})
    add_test( ${t}_BLOCKING ${CMAKE_CURRENT_BINARY_DIR}/${t}_BLOCKING )
    set_tests_properties ( ${t}_BLOCKING PROPERTIES TIMEOUT 180)

endforeach( t )

# tests with special compilation parameters
# set_target_properties(test_scheduling2_NONBLOCKING PROPERTIES
#     COMPILE_DEFINITIONS LB_CALLBACK)
# set_target_properties(test_scheduling2_BLOCKING PROPERTIES
# 	  COMPILE_DEFINITIONS LB_CALLBACK)