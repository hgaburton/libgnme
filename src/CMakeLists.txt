set(SRC
    slater/slater_uscf.C
    utils/eri_ao2mo.C
    utils/linalg.C
    utils/lowdin_pair.C
    utils/utils.C
    wick/wick.C
    wick/wick_overlap.C
    wick/wick_one_body.C
    wick/wick_two_body.C
    wick/wick_1rdm.C
)

add_library(gnme STATIC ${SRC})
install(TARGETS gnme DESTINATION lib)
