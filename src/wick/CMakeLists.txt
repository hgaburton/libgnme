set(SRC
    noci_kernels.C
    noci_wick.C
    noci_wick_overlap.C
    noci_wick_one_body.C
    noci_wick_two_body.C
)

add_library(noci_wick ${SRC})
target_include_directories(noci_wick PUBLIC "${LIBNOCI_SOURCE_DIR}")
target_link_libraries(noci_wick gscf_fock)

install(TARGETS noci_wick DESTINATION lib)
