set(SRC
    utils/linalg.C
    utils/utils.C
    utils/lowdin_pair.C
    wick/wick.C
    wick/wick_overlap.C
    wick/wick_one_body.C
    wick/wick_two_body.C
    wick/wick_1rdm.C
)

add_library(gnme STATIC ${SRC})
install(TARGETS gnme DESTINATION lib)
