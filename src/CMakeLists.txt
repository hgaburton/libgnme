set(SRC
    utils/utils.C
    utils/lowdin_pair.C
    wick/wick.C
    wick/wick_overlap.C
    wick/wick_one_body.C
    wick/wick_two_body.C
)

add_library(nome SHARED ${SRC})
install(TARGETS nome DESTINATION lib)
