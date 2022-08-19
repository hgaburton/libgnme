#
#   This script sets up header-only use of Armadillo without regard for how
#   Armadillo was configured and installed on the system. It also forces our
#   choice of BLAS and LAPACK libraries
#
#   By including this script, the user will have initialized an interface
#   library called armadillo with all the include paths, compiler definitions,
#   and linking flags and depended libraries.
#
#   Use:
#   add_library(foo ...)
#   target_link_libraries(foo ... armadillo)
#

if(NOT TARGET armadillo)

if(ARMADILLO_PATH)
    find_path(ARMADILLO_INCLUDE_DIRS armadillo PATHS "${ARMADILLO_PATH}"
        PATH_SUFFIXES include usr/include usr/local/include NO_DEFAULT_PATH)
else(ARMADILLO_PATH)
    find_path(ARMADILLO_INCLUDE_DIRS armadillo)
endif(ARMADILLO_PATH)
if(ARMADILLO_INCLUDE_DIRS)
    add_library(armadillo INTERFACE)
    set(ARMADILLO_COMPILE_DEFINITIONS
        ARMA_DONT_USE_WRAPPER ARMA_USE_NEWARP ARMA_DONT_USE_ATLAS
        ARMA_DONT_USE_SUPERLU ARMA_DONT_USE_HDF5
        ARMA_DONT_PRINT_CXX11_WARNING
	ARMA_DONT_USE_OPENMP ARMA_64BIT_WORD
        $<$<NOT:$<CONFIG:Debug>>:ARMA_NO_DEBUG>)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        list(APPEND ARMADILLO_COMPILE_DEFINITIONS
            ARMA_ALLOW_FAKE_GCC ARMA_ALLOW_FAKE_CLANG)
    endif()
    if(BLAS_FOUND)
        list(APPEND ARMADILLO_COMPILE_DEFINITIONS ARMA_USE_BLAS)
    else(BLAS_FOUND)
        list(APPEND ARMADILLO_COMPILE_DEFINITIONS ARMA_DONT_USE_BLAS)
    endif(BLAS_FOUND)
    if(LAPACK_FOUND)
        list(APPEND ARMADILLO_COMPILE_DEFINITIONS ARMA_USE_LAPACK)
    else(LAPACK_FOUND)
        list(APPEND ARMADILLO_COMPILE_DEFINITIONS ARMA_DONT_USE_LAPACK)
    endif(LAPACK_FOUND)
    target_compile_definitions(armadillo INTERFACE
        ${ARMADILLO_COMPILE_DEFINITIONS})
    target_include_directories(armadillo INTERFACE "${ARMADILLO_INCLUDE_DIRS}")
    if(BLAS_FOUND)
        target_link_libraries(armadillo INTERFACE
            ${BLAS_LINKER_FLAGS} ${BLAS_LIBRARIES})
    endif(BLAS_FOUND)
    if(LAPACK_FOUND)
        target_link_libraries(armadillo INTERFACE
            ${LAPACK_LINKER_FLAGS} ${LAPACK_LIBRARIES})
    endif(LAPACK_FOUND)
    set(Armadillo_FOUND TRUE)
    message(STATUS "Found Armadillo: ${ARMADILLO_INCLUDE_DIRS}")
else(ARMADILLO_INCLUDE_DIRS)
    message(ERROR "Cannot find Armadillo")
endif(ARMADILLO_INCLUDE_DIRS)

endif(NOT TARGET armadillo)
