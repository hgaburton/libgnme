set(SRC
    utils/utils.C
    utils/lowdin_pair.C
)

add_library(nome SHARED ${SRC})
install(TARGETS nome DESTINATION lib)
