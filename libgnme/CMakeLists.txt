add_subdirectory(utils)
add_subdirectory(wick)
add_subdirectory(slater)

add_library(gnme INTERFACE)
target_link_libraries(gnme INTERFACE gnme_utils gnme_wick gnme_slater armadillo)
