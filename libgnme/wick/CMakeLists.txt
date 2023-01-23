set(SRC
    wick_orbitals.C
    wick_rscf_rdm.C
    wick_rscf.C
    wick_rscf_overlap.C
    wick_rscf_one_body.C
    wick_rscf_two_body.C
    wick_uscf.C
    wick_uscf_overlap.C
    wick_uscf_one_body.C
    wick_uscf_two_body.C
    wick_uscf_rdm.C
)

add_library(gnme_wick ${SRC})
target_include_directories(gnme_wick PUBLIC "${$LIBGNME_SOURCE_DIR}")
target_link_libraries(gnme_wick gnme_utils)

install(TARGETS gnme_wick DESTINATION lib)
