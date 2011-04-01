set ( PBZIP2SRCS
        pbzip2.cpp
    )

set ( PBZIP2_FFSRCS
        pbzip2_ff.cpp
    )

add_executable ( pbzip2 ${PBZIP2SRCS} )
add_executable ( pbzip2_ff ${PBZIP2_FFSRCS} )

include_directories ( ${PROJECT_SOURCE_DIR}
                      ${PROJECT_SOURCE_DIR}/atomic/
                    )

target_link_libraries ( pbzip2_ff bz2 ${CMAKE_THREAD_LIBS_INIT}  )
target_link_libraries ( pbzip2 bz2 ${CMAKE_THREAD_LIBS_INIT}  )

set_target_properties ( pbzip2      PROPERTIES COMPILE_FLAGS "-D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64" )
set_target_properties ( pbzip2_ff   PROPERTIES COMPILE_FLAGS "-D_LARGEFILE64_SOURCE -D_FILE_OFFSET_BITS=64" )

