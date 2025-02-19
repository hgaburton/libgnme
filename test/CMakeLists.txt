set(TESTS 
    casci_uscf_coupling
    casci_rscf_coupling
    casci_rscf_coupling_split
    wick_orbitals_test
    wick_uscf_one_body_test
    wick_uscf_two_body_test
    wick_rscf_one_body_test
    wick_rscf_two_body_test
)

foreach(TEST ${TESTS})
    set(TGT "gnme_${TEST}")
    list(APPEND TEST_TGTS ${TGT})
    add_executable(${TGT} ${TEST}.C)
    target_link_libraries(${TGT} gnme armadillo)
    if(WITH_OPENMP)
        if(OpenMP_CXX_FOUND)
            target_link_libraries(${TGT} OpenMP::OpenMP_CXX)
        endif()
    endif(WITH_OPENMP)
    add_test(NAME "gnme/${TEST}" COMMAND ${TGT} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
add_custom_target(gnme_tests DEPENDS ${TEST_TGTS})
