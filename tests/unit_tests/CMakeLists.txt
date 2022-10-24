FIND_PACKAGE(GTest REQUIRED)
FIND_PACKAGE(cuPoly REQUIRED)

INCLUDE_DIRECTORIES(${GTEST_INCLUDE_DIRS})

#
# TEST
#
ENABLE_TESTING()

CUDA_ADD_EXECUTABLE(
    cupoly_bfv_test
    test_bfv_mode.cu
)

TARGET_LINK_LIBRARIES(
    cupoly_bfv_test
    ${CUDA_curand_LIBRARY}
    ${GTEST_BOTH_LIBRARIES}
    -lpthread
    cupolybfv
)

CUDA_ADD_EXECUTABLE(
    cupoly_ckks_test
    test_ckks_mode.cu
)

TARGET_LINK_LIBRARIES(
    cupoly_ckks_test
    ${CUDA_curand_LIBRARY}
    ${GTEST_BOTH_LIBRARIES}
    -lpthread
    cupolyckks
)

# GTEST_ADD_TESTS(
# 	TARGET cupoly_test)