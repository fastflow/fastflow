# Check headers
include(CheckIncludeFile)
CHECK_INCLUDE_FILE(malloc.h HAVE_MALLOC_H)
CHECK_INCLUDE_FILE(numa.h HAVE_NUMA_H)
CHECK_INCLUDE_FILE(stdint.h  HAVE_STDINT_H)
CHECK_INCLUDE_FILE(pthread.h  HAVE_PTHREAD_H)
CHECK_INCLUDE_FILE(omp.h HAVE_OMP_H)

# Compiler type
SET(HAS_GCC ${CMAKE_COMPILER_IS_GNUCC})
SET(HAS_GXX ${CMAKE_COMPILER_IS_GNUCCXX})
SET(HAS_CLANGXX ${FFCM_HAS_CLANGXX})
SET(HAS_MSVC ${CMAKE_COMPILER_IS_GNUCC})

# Check compiler features
include(${PROJECT_SOURCE_DIR}/cmake.modules/cmake_cxx11/CheckCXX11Features.cmake)
foreach(flag ${CXX11_FEATURE_LIST})
    message(STATUS  "Cxx11 detection: ${flag}")
endforeach(flag ${CXX11_FEATURE_LIST})



CONFIGURE_FILE(${CMAKE_CURRENT_SOURCE_DIR}/cmake.modules/ffconfig.h.in ${CMAKE_CURRENT_SOURCE_DIR}/cmake.modules/ffconfig.h)