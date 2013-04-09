#cmake_minimum_required(VERSION 2.6)
#project( denoiser )

#find_package (Threads)

set (SRCS driver.cpp parameters.cpp)

#SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules ) #comment on linux

#ff
include_directories (
    ${PROJECT_SOURCE_DIR}
#    ${CMAKE_CURRENT_SOURCE_DIR} #useless?
    )

#OpenCL
include_directories ("/usr/local/cuda/include") #OpenCL #comment on mac
FIND_LIBRARY( OpenCL_LIB OpenCL /usr/lib64) #comment on mac
#set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL") #comment on linux

#flag for OpenCL versions: SPD_OCL

add_definitions( -DKERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernel_ocl/fuy.cl" )

add_executable (denoiser ${SRCS})
set_target_properties (denoiser PROPERTIES COMPILE_DEFINITIONS "BITMAP;TIME;SPD_OCL")
target_link_libraries (denoiser ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES (denoiser ${OpenCV_LIBS} )
target_link_libraries (denoiser ${OpenCL_LIB} ) #comment on mac

add_executable (denoiser_video ${SRCS})
set_target_properties (denoiser_video PROPERTIES COMPILE_DEFINITIONS "FROMFILE;TIME;SPD_OCL")
target_link_libraries (denoiser_video ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES( denoiser_video ${OpenCV_LIBS} )
target_link_libraries (denoiser_video ${OpenCL_LIB} ) #comment on mac

add_executable (denoiser_camera ${SRCS})
set_target_properties (denoiser_camera PROPERTIES COMPILE_DEFINITIONS "FROMCAMERA;TIME;SPD_OCL")
target_link_libraries (denoiser_camera ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES( denoiser_camera ${OpenCV_LIBS} )
target_link_libraries (denoiser_camera ${OpenCL_LIB} ) #comment on mac

add_executable (denoiser_color ${SRCS})
set_target_properties (denoiser_color PROPERTIES COMPILE_DEFINITIONS "BITMAPCOLOR;TIME;SPD_OCL")
target_link_libraries (denoiser_color ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES (denoiser_color ${OpenCV_LIBS} )
target_link_libraries (denoiser_color ${OpenCL_LIB} ) #comment on mac
