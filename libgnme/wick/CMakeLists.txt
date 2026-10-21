set(SRC 
    wick_base_overlap.C
    wick_base_rdm.C
    wick_orbitals.C
    wick_kernels.C
    wick_matrix.C
    wick_pair.C
    wick_rscf.C
    wick_uscf.C
)

# Packed AVX2 and AVX-512 kernels are compiled separately and selected at run time, so the 
# library does not require either instruction set
if(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64" AND CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    list(APPEND SRC wick_kernels_avx2.C wick_kernels_avx512.C)
    set_source_files_properties(wick_kernels_avx2.C PROPERTIES COMPILE_OPTIONS "-mavx2;-mfma")
    set_source_files_properties(wick_kernels_avx512.C PROPERTIES COMPILE_OPTIONS "-mavx512f")
    set_source_files_properties(wick_kernels.C PROPERTIES COMPILE_DEFINITIONS LIBGNME_X86_SIMD)
endif()

add_library(gnme_wick ${SRC})
target_include_directories(gnme_wick PUBLIC "${$LIBGNME_SOURCE_DIR}")
target_link_libraries(gnme_wick gnme_utils "${BLAS_LIBRARIES}" )
target_link_libraries(gnme_wick gnme_utils "${LAPACK_LIBRARIES}" )
include_directories("${CBLAS_LIBRARIES}")

install(TARGETS gnme_wick DESTINATION lib)
