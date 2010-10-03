set ( QSFF_SRCS ff_qs.cpp )
set ( QS_SRCS qs.cpp )

include_directories (
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
    ${CURRENT_SOURCE_DIR}
    )
add_executable ( qs ${QS_SRCS} )
add_executable ( ff_qs ${QSFF_SRCS} )

target_link_libraries ( qs )
target_link_libraries ( ff_qs pthread )

set_target_properties ( qs          PROPERTIES COMPILE_FLAGS "-O3 -Wall " )
set_target_properties ( ff_qs       PROPERTIES COMPILE_FLAGS "-O3 -Wall " )

