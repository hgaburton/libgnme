set(TESTS
    wick_two_body_timing
)

foreach(TEST ${TESTS})
    set(TGT "gnme_${TEST}")
    list(APPEND TEST_TGTS ${TGT})
    add_executable(${TGT} ${TEST}.C)
    target_link_libraries(${TGT} gnme armadillo)
    add_test(NAME "gnme/${TEST}" COMMAND ${TGT} WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
add_custom_target(gnme_tests DEPENDS ${TEST_TGTS})
