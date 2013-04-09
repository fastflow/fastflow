find_package (X11 REQUIRED)

set( MANSRC         mandel_ff.cpp )
set( MANSEQSRC      mandel_seq.c )
set( MANMEMALLSRC   mandel_ff_mem_all.cpp )
set( MANPTSRC       mandel_pt.cpp )

set( MARX2SRC       marX2.c )

# ${X11_LIBRARIES} has been used instead of X11 to prevent a bug on
# MacOS in the linking of -lx11 that requires -L ...
 
set( LINKS          marX2 pthread  m ${X11_LIBRARIES})
set( TARGETS        mandel_ff mandel_seq mandel_ff_mem_all mandel_pt )


include_directories( 
        ${X11_INCLUDE_DIR}
        ${PROJECT_SOURCE_DIR}
        ${CMAKE_CURRENT_SOURCE_DIR} )
add_library( marX2 STATIC ${MARX2SRC} )

if(COMMAND cmake_policy)
    cmake_policy(SET CMP0003 NEW)
endif(COMMAND cmake_policy)

add_executable( mandel_ff           ${MANSRC} )
add_executable( mandel_seq          ${MANSEQSRC} )
add_executable( mandel_ff_mem_all   ${MANMEMALLSRC} )
add_executable( mandel_pt           ${MANPTSRC} )


foreach( f ${TARGETS} )
    target_link_libraries( ${f} ${LINKS}  )
endforeach(f)
