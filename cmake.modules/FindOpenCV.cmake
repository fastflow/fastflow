##----------------------------------------------------------
# Temporary FindOpenCV.cmake
# Marco Aldinucci - 4 Nov 2012
# Just a sketch - to be substituted if/when a really working FindOpenCV.cmake
# will be delivered
# Don't try to do anything else than finding a OpenCV configuration
# if cannot be found just fails. OpenCV webpage cmake simply does not work
##----------------------------------------------------------

IF(DEFINED ENV{OpenCV_DIR})
  SET(OpenCV_DIR $ENV{OpenCV_DIR})
  MESSAGE(STATUS "OpenCV_DIR read from environment: ${OpenCV_DIR}")
ELSE()
find_path(OpenCV_DIR "OpenCVConfig.cmake" HINTS /usr/share/opencv/ /opt/local/lib/cmake /opencv/build DOC "Root directory of OpenCV") 
  message(STATUS "Looking for OpenCVConfig.cmake, found in: ${OpenCV_DIR}")
ENDIF()


##====================================================
## Find OpenCV libraries
##----------------------------------------------------
	
if(EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
          ## Include the standard CMake script
          include("${OpenCV_DIR}/OpenCVConfig.cmake")
	  message(STATUS "Found OpenCVConfig.cmake in ${OpenCV_DIR} (found version ${OpenCV_VERSION})")  
	  ## Search for a specific version
          set(CVLIB_SUFFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
          set(OpenCV_FOUND true)
elseif(EXISTS "${OpenCV_DIR}/share/OpenCV/OpenCVConfig.cmake")
	  ## Include the standard CMake script
          include("${OpenCV_DIR}/share/OpenCV/OpenCVConfig.cmake")
	  message(STATUS "Found OpenCVConfig.cmake in ${OpenCV_DIR}/share/OpenCV (found version ${OpenCV_VERSION})")  
	  ## Search for a specific version
          set(CVLIB_SUFFIX "${OpenCV_VERSION_MAJOR}${OpenCV_VERSION_MINOR}${OpenCV_VERSION_PATCH}")
          set(OpenCV_FOUND true)
#Otherwise fails.
else()
	set(OpenCV_FOUND false)
endif(EXISTS "${OpenCV_DIR}/OpenCVConfig.cmake")
    
##----------------------------------------------------
if(NOT OpenCV_FOUND)
        # make FIND_PACKAGE friendly
        if(NOT OpenCV_FIND_QUIETLY)
                if(OpenCV_FIND_REQUIRED)
                        message(FATAL_ERROR "OpenCV required but some headers or libs not found. ${ERR_MSG}")
                else(OpenCV_FIND_REQUIRED)
                        message(STATUS "WARNING: OpenCV was not found. ${ERR_MSG}")
                endif(OpenCV_FIND_REQUIRED)
        endif(NOT OpenCV_FIND_QUIETLY)
endif(NOT OpenCV_FOUND)
##====================================================


##====================================================
## Backward compatibility
##----------------------------------------------------
if(OpenCV_FOUND)
        option(OpenCV_BACKWARD_COMPA "Add some variable to make this script compatible with the other version of FindOpenCV.cmake" false)
        if(OpenCV_BACKWARD_COMPA)
                find_path(OpenCV_INCLUDE_DIRS "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "Include directory") 
                find_path(OpenCV_INCLUDE_DIR "cv.h" PATHS "${OpenCV_DIR}" PATH_SUFFIXES "include" "include/opencv" DOC "Include directory")
                set(OpenCV_LIBRARIES "${OpenCV_LIBS}" CACHE STRING "" FORCE)
        endif(OpenCV_BACKWARD_COMPA)
endif(OpenCV_FOUND)
##====================================================
