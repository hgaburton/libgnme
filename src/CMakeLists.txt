set(SRC
    utils/linalg.C
    utils/utils.C
    utils/lowdin_pair.C
    wick/wick.C
    wick/wick_overlap.C
    wick/wick_one_body.C
    wick/wick_two_body.C
    cat.C
)

add_library(nome STATIC ${SRC})
install(TARGETS nome DESTINATION lib)
