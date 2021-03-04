# Minimum version of CMake is 3.12
cmake_minimum_required(VERSION 3.12 FATAL_ERROR)

# Minimum CXX standards
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Define project
project(libnome VERSION 0.0.1 LANGUAGES CXX)

# Setup CMake
message(STATUS "CMake ${CMAKE_VERSION}")
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake" ${CMAKE_MODULE_PATH})

# Setup OpenMP
if(WITH_OPENMP)
    find_package(OpenMP REQUIRED)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set(CMAKE_Fortran_FLAGS "${CMAKE_Fortran_FLAGS} ${OpenMP_Fortran_FLAGS}")
    message(STATUS "OpenMP is enabled")
endif(WITH_OPENMP)

# Setup Armadillo paths
set(ARMADILLO_PATH "${PROJECT_SOURCE_DIR}/external/armadillo-10.1.2")
find_package(BLAS)
find_package(LAPACK)
include(cmake/SetupArmadillo.cmake)
set(CMAKE_MODULE_PATH "${PROJECT_SOURCE_DIR}/cmake")
message(STATUS "ARMADILLO: ${ARMADILLO_INCLUDE_DIRS} ${ARMADILLO_LIBRARIES}")
include_directories(${ARMADILLO_INCLUDE_DIRS})

add_subdirectory(src)

# Setup Doxygen
find_package(Doxygen 1.8)
if(DOXYGEN_FOUND)
    configure_file(
        "${CMAKE_CURRENT_SOURCE_DIR}/docs/Doxyfile.in"
        "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile" @ONLY)
    add_custom_target(nome_docs
        "${DOXYGEN_EXECUTABLE}" "${CMAKE_CURRENT_BINARY_DIR}/Doxyfile"
        WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
        COMMENT "Doxygen documentation" VERBATIM)
endif(DOXYGEN_FOUND)

target_link_libraries(nome ${ARMADILLO_LIBRARIES})

if(CMAKE_RUN_FROM_CONFIGURE)
    message(STATUS "Install location: ${CMAKE_INSTALL_PREFIX}")
    message(STATUS "-----------------------------------------")
    message(STATUS " > Configure complete!                   ")
    message(STATUS " > Now run make inside build/ directory  ")
    message(STATUS "-----------------------------------------")
    unset(CMAKE_RUN_FROM_CONFIGURE CACHE)
endif(CMAKE_RUN_FROM_CONFIGURE)
