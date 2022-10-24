PROJECT(${CMAKE_PROJECT_NAME}_tests)

# Clone Google Test
SET(GOOGLETEST_DIR ${CMAKE_SOURCE_DIR}/tests/lib/googletest)
if(NOT EXISTS ${GOOGLETEST_DIR})
    FIND_PACKAGE(Git REQUIRED)
    EXECUTE_PROCESS(
        COMMAND ${GIT_EXECUTABLE} clone https://github.com/google/googletest ${GOOGLETEST_DIR}
    )
endif()

# add_subdirectory(lib/googletest)
ADD_SUBDIRECTORY(unit_tests)
