set( CHOLSRC cholesky.cpp )
set( BCHOLSRC blockCholesky.cpp )

include_directories(    ${PROJECT_SOURCE_DIR}/
                        ${CMAKE_CURRENT_SOURCE_DIR}
                        ${CMAKE_CURRENT_SOURCE_DIR}/include
                    )

add_executable( cholesky ${CHOLSRC} )
add_executable( blockCholesky ${BCHOLSRC} )

target_link_libraries( cholesky pthread )
target_link_libraries( blockCholesky pthread )

set_target_properties( cholesky         PROPERTIES COMPILE_FLAGS "-O3 -Wall" )
set_target_properties( blockCholesky    PROPERTIES COMPILE_FLAGS "-O3 -Wall" )
