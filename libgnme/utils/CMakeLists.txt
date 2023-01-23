set(SRC
    bitset_tools.C
    eri_ao2mo.C  
    linalg.C  
    lowdin_pair.C  
    utils.C
)

add_library(gnme_utils ${SRC})
target_include_directories(gnme_utils PUBLIC "${LIBGNME_SOURCE_DIR}")

install(TARGETS gnme_utils DESTINATION lib)
