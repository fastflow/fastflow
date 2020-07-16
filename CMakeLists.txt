cmake_minimum_required(VERSION 3.11)

project(
  FastFLow
  VERSION 3
  DESCRIPTION "FastFlow pattern-based parallel programming framework"
  LANGUAGES CXX)

# Only do these if this is the main project, and not if it is included through
# add_subdirectory
if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  # Define cmake cached variables
  option(BUILD_TESTING "Activate the tests building" OFF)

  # Set the compiler standad to c++17
  set(CMAKE_CXX_STANDARD 17)
  set(CMAKE_CXX_STANDARD_REQUIRED ON)
  set(CMAKE_CXX_EXTENSIONS OFF) # Let's ensure -std=c++xx instead of -std=g++xx

  # Set warnings
  if(NOT MSVC)
    add_compile_options(-Wall -Wextra)
  endif()

  # Add tests
  if(BUILD_TESTING)
    message(STATUS "Configuring Tests")
    include(CTest)
    add_subdirectory(tests)
  else()
    # If tests will not be compiled show a status message during configure phase
    # and a red warning during build
    message(STATUS "Skiping Tests (pass -DBUILD_TESTING=ON to activate)")
    add_custom_target(
      warning ALL
      ${CMAKE_COMMAND}
      -E
      cmake_echo_color
      --red
      "Nothing to compile by default since FastFlow is an header-only library."
      "To compile the tests, please follow the instructions contained in README.md."
      "To install the library in the selected folder, type \\\"make install\\\"."
      "The install target will also execute the \\\"mapping_string.sh\\\" script automatically."
    )
  endif()

endif()

# Create the Fastflow library
add_library(fastflow INTERFACE)
target_include_directories(
  fastflow INTERFACE $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                     $<INSTALL_INTERFACE:include>)

# Install all the FastFlow headers
install(DIRECTORY ff/ DESTINATION include/ff)

# Execute the Mapping script if lstopo is founded
find_program(LSTOPO lstopo)
if(NOT LSTOPO)
  message(
    WARNING "'lstopo' program not found"
            "(Install the 'hwloc' package and run again the installation)")
  message(STATUS "Mapping configuration skipped")
  install(FILES ${CMAKE_CURRENT_SOURCE_DIR}/ff/config.hpp
          DESTINATION include/ff)
else()
  message(STATUS "'lstopo' program found")
  message(STATUS "Mapping configuration will run")

  # Execute the mapping script and update the config.hpp file
  add_custom_command(
    OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/ff/config.hpp"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_SOURCE_DIR}/ff/config.hpp"
            "${CMAKE_CURRENT_BINARY_DIR}/config.hpp"
    COMMAND ${CMAKE_COMMAND} -E echo "yes" "|"
            "${CMAKE_CURRENT_SOURCE_DIR}/ff/mapping_string.sh"
    COMMAND ${CMAKE_COMMAND} -E copy "${CMAKE_CURRENT_BINARY_DIR}/config.hpp"
            "${CMAKE_CURRENT_BINARY_DIR}/ff/config.hpp"
    COMMAND ${CMAKE_COMMAND} -E remove "${CMAKE_CURRENT_BINARY_DIR}/config.hpp")

  # Add a custom target and install the modified config.hpp
  add_custom_target(generate_mapping ALL
                    DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/ff/config.hpp")

  install(FILES ${CMAKE_CURRENT_BINARY_DIR}/ff/config.hpp
          DESTINATION include/ff)
endif()
