find_package ( Qt4 REQUIRED )
include( ${QT_USE_FILE} )

set ( MANDSRCS
    main.cpp
    mandelbrotwidget.cpp
    renderthread.cpp
)

set ( MANDHDRS
        mandelbrotwidget.h
        renderthread.h
)

include_directories( 
    ${QT_INCLUDE_DIR}
    ${CURRENT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}
    ${PROJECT_SOURCE_DIR}/atomic
)

QT4_WRAP_CPP( MOC_SRCS ${MANDHDRS} )
 
add_executable ( mandelbrot_ff ${MANDSRCS} ${MOC_SRCS} )
target_link_libraries ( mandelbrot_ff ${QT_LIBRARIES} )
