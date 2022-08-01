set(SRC
    slater_uscf.C
    slater_rscf.C
)

add_library(gnme_slater ${SRC})
target_include_directories(gnme_slater PUBLIC "${$LIBGNME_SOURCE_DIR}")
target_link_libraries(gnme_slater gnme_utils)

install(TARGETS gnme_slater DESTINATION lib)
