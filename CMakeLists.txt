# Minimum version of CMake is 3.12
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Minimum CXX standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project
project(LIBGNME VERSION 0.0.1 LANGUAGES CXX)
message(STATUS ${LIBGNME_SOURCE_DIR})

# Setup CMake
message(STATUS "CMake ${CMAKE_VERSION}")
set(CMAKE_MODULE_PATH "${LIBGNME_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Setup OpenMP
if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    message(STATUS "OpenMP is enabled")
endif(WITH_OPENMP)

# Look for linear algebra libraries
find_package(BLAS)
find_package(LAPACK)

# Setup Armadillo paths
set(ARMADILLO_PATH "${LIBGNME_SOURCE_DIR}/external/armadillo-10.1.2")
include(cmake/SetupArmadillo.cmake)
include_directories($<TARGET_PROPERTY:armadillo,INTERFACE_INCLUDE_DIRECTORIES>)
set_property(DIRECTORY . APPEND PROPERTY COMPILE_DEFINITIONS
    $<TARGET_PROPERTY:armadillo,INTERFACE_COMPILE_DEFINITIONS>)

find_package(Doxygen 1.8)
enable_testing()

set(CMAKE_MODULE_PATH "${LIBGNME_SOURCE_DIR}/cmake")
#message(STATUS "ARMADILLO: ${ARMADILLO_INCLUDE_DIRS} ${ARMADILLO_LIBRARIES}")
#include_directories(${ARMADILLO_INCLUDE_DIRS})

add_subdirectory(libgnme)
add_subdirectory(test)

# Setup Doxygen
if(DOXYGEN_FOUND)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)
    add_custom_target(gnme_docs
        "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Doxygen documentation" VERBATIM)
endif(DOXYGEN_FOUND)

if(CMAKE_RUN_FROM_CONFIGURE)
    message(STATUS "Install location: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "-----------------------------------------")
    message(STATUS " > Configure complete!                   ")
    message(STATUS " > Now run make inside build/ directory  ")
    message(STATUS "-----------------------------------------")
    unset(CMAKE_RUN_FROM_CONFIGURE CACHE)
endif(CMAKE_RUN_FROM_CONFIGURE)
