add_subdirectory(utils)
add_subdirectory(wick)
add_subdirectory(slater)

add_library(gnme INTERFACE)
target_link_libraries(gnme INTERFACE gnme_utils gnme_wick gnme_slater armadillo)
# specify open MP flags

if(WITH_OPENMP)
  find_package(OpenMP REQUIRED)
  if(OpenMP_CXX_FOUND)
      target_link_libraries(gnme INTERFACE OpenMP::OpenMP_CXX)
  endif()
endif(WITH_OPENMP)

