set(SRC 
    one_body_rscf.C
    one_body_uscf.C
    two_body_rscf.C
    two_body_uscf.C
    wick_base.C
    wick_orbitals.C
    wick_rscf.C
    wick_uscf.C
)

add_library(gnme_wick ${SRC})
target_include_directories(gnme_wick PUBLIC "${$LIBGNME_SOURCE_DIR}")
target_link_libraries(gnme_wick gnme_utils "${BLAS_LIBRARIES}" )
target_link_libraries(gnme_wick gnme_utils "${LAPACK_LIBRARIES}" )
include_directories("${CBLAS_LIBRARIES}")

install(TARGETS gnme_wick DESTINATION lib)
