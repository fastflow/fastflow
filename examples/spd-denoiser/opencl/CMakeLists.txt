set (SRCS driver.cpp parameters.cpp)

if(APPLE)
	SET( CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake.modules ) 
endif(APPLE)

include_directories (${PROJECT_SOURCE_DIR} )

#FF version

add_executable (denoiser ${SRCS})
set_target_properties (denoiser PROPERTIES COMPILE_DEFINITIONS "BITMAP;TIME")
target_link_libraries (denoiser ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES (denoiser ${OpenCV_LIBS} )

add_executable (denoiser_video ${SRCS})
set_target_properties (denoiser_video PROPERTIES COMPILE_DEFINITIONS "FROMFILE;TIME")
target_link_libraries (denoiser_video ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES( denoiser_video ${OpenCV_LIBS} )

add_executable (denoiser_camera ${SRCS})
set_target_properties (denoiser_camera PROPERTIES COMPILE_DEFINITIONS "FROMCAMERA;TIME")
target_link_libraries (denoiser_camera ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES( denoiser_camera ${OpenCV_LIBS} )

add_executable (denoiser_color ${SRCS})
set_target_properties (denoiser_color PROPERTIES COMPILE_DEFINITIONS "BITMAPCOLOR;TIME")
target_link_libraries (denoiser_color ${CMAKE_THREAD_LIBS_INIT})
TARGET_LINK_LIBRARIES (denoiser_color ${OpenCV_LIBS} )

#Tests FF
add_test( denoiser ${CMAKE_CURRENT_BINARY_DIR}/denoiser -v ${CMAKE_CURRENT_SOURCE_DIR}/test/lena512_noise50.bmp)
add_test( denoiser_color ${CMAKE_CURRENT_BINARY_DIR}/denoiser_color -v ${CMAKE_CURRENT_SOURCE_DIR}/test/lena_c_40.bmp)

#OpenCL Version (flag for OpenCL versions: SPD_OCL)

find_package(OpenCL)
  
if (OPENCL_FOUND) 
  
	include_directories(${OPENCL_INCLUDE_DIRS})
	add_definitions( -DKERNEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/kernel_ocl/fuy.cl" )
	
	if(APPLE)
		set(CMAKE_EXE_LINKER_FLAGS "-framework OpenCL") 
	endif(APPLE)
	
	add_executable (denoiser_ocl ${SRCS})
	set_target_properties (denoiser_ocl PROPERTIES COMPILE_DEFINITIONS "BITMAP;TIME;SPD_OCL")
	target_link_libraries (denoiser_ocl ${CMAKE_THREAD_LIBS_INIT})
	TARGET_LINK_LIBRARIES (denoiser_ocl ${OpenCV_LIBS} )
	
	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		target_link_libraries (denoiser_ocl ${OPENCL_LIBRARIES} )
	endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")	
	
	add_executable (denoiser_video_ocl ${SRCS})
	set_target_properties (denoiser_video_ocl PROPERTIES COMPILE_DEFINITIONS "FROMFILE;TIME;SPD_OCL")
	target_link_libraries (denoiser_video_ocl ${CMAKE_THREAD_LIBS_INIT})
	TARGET_LINK_LIBRARIES( denoiser_video_ocl ${OpenCV_LIBS} )
	
	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		target_link_libraries (denoiser_video_ocl ${OPENCL_LIBRARIES} )
	endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	
	add_executable (denoiser_camera_ocl ${SRCS})
	set_target_properties (denoiser_camera_ocl PROPERTIES COMPILE_DEFINITIONS "FROMCAMERA;TIME;SPD_OCL")
	target_link_libraries (denoiser_camera_ocl ${CMAKE_THREAD_LIBS_INIT})
	TARGET_LINK_LIBRARIES( denoiser_camera_ocl ${OpenCV_LIBS} )
	
	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		target_link_libraries (denoiser_camera_ocl ${OPENCL_LIBRARIES} )
	endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	
	add_executable (denoiser_color_ocl ${SRCS})
	set_target_properties (denoiser_color_ocl PROPERTIES COMPILE_DEFINITIONS "BITMAPCOLOR;TIME;SPD_OCL")
	target_link_libraries (denoiser_color_ocl ${CMAKE_THREAD_LIBS_INIT})
	TARGET_LINK_LIBRARIES (denoiser_color_ocl ${OpenCV_LIBS} )
	
	if(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
		target_link_libraries (denoiser_color_ocl ${OPENCL_LIBRARIES} )
	endif(${CMAKE_SYSTEM_NAME} MATCHES "Linux")
	
	#Tests OpenCL
	if(OPTIRUN)
		MESSAGE(STATUS "Using ${OPTIRUN} to test GPU-enabled tests: denoiser_color")
	endif(OPTIRUN)
	
	add_test( denoiser_ocl ${OPTIRUN} ${CMAKE_CURRENT_BINARY_DIR}/denoiser_ocl -v ${CMAKE_CURRENT_SOURCE_DIR}/test/lena512_noise50.bmp)
	add_test( denoiser_color_ocl ${OPTIRUN} ${CMAKE_CURRENT_BINARY_DIR}/denoiser_color_ocl -v ${CMAKE_CURRENT_SOURCE_DIR}/test/lena_c_40.bmp)
	
else (OPENCL_FOUND)
	message (WARNING "OpenCL not found: Skipping denoiser OpenCL example")
endif(OPENCL_FOUND)