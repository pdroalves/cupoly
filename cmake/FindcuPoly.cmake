# Try to find the cuPoly library
# CUPOLY_FOUND - system has CUPOLY lib
# CUPOLY_INCLUDE_DIR - the CUPOLY include directory
# CUPOLY_LIBRARIES - Libraries needed to use CUPOLY

if (CUPOLY_INCLUDE_DIR AND CUPOLY_LIBRARIES)
    # Already in cache, be silent
    set(CUPOLY_FIND_QUIETLY TRUE)
endif (CUPOLY_INCLUDE_DIR AND CUPOLY_LIBRARIES)

FIND_PATH(CUPOLY_INCLUDE_DIR NAMES cuPoly/settings.h )
FIND_LIBRARY(CUPOLY_LIBRARIES NAMES cupolybfv libcupolybfv )

set( CUPOLY_VERSION_H "../../include/cuPoly/cuPolyConfig.h" )
file( READ "${CUPOLY_VERSION_H}" CUPOLY_VERSION_H_CONTENTS )

# Load version
# Major
string(REGEX MATCH
	  "define[ \t]cuPoly_VERSION_MAJOR[ \t]+([0-9]+)" _cuPoly_VERSION_MAJOR
	  "${CUPOLY_VERSION_H_CONTENTS}")
if(_cuPoly_VERSION_MAJOR)
	set(cuPoly_VERSION_MAJOR "${CMAKE_MATCH_1}")
endif()

# Minor
string(REGEX MATCH
	  "define[ \t]cuPoly_VERSION_MINOR[ \t]+([0-9]+)" _cuPoly_VERSION_MINOR
	  "${CUPOLY_VERSION_H_CONTENTS}")
if(_cuPoly_VERSION_MINOR)
	set(cuPoly_VERSION_MINOR "${CMAKE_MATCH_1}")
endif()

# Patch
string(REGEX MATCH
	  "define[ \t]cuPoly_VERSION_PATCH[ \t]+([0-9]+)" _cuPoly_VERSION_PATCH
	  "${CUPOLY_VERSION_H_CONTENTS}")
if(_cuPoly_VERSION_PATCH)
	set(cuPoly_VERSION_PATCH "${CMAKE_MATCH_1}")
endif()

# Tweak
string(REGEX MATCH
	  "define[ \t]cuPoly_VERSION_TWEAK[ \t]+([0-9]+)" _cuPoly_VERSION_TWEAK
	  "${CUPOLY_VERSION_H_CONTENTS}")
if(_cuPoly_VERSION_TWEAK)
	set(cuPoly_VERSION_TWEAK "${CMAKE_MATCH_1}")
endif()

SET( CUPOLY_VERSION "v" ${cuPoly_VERSION_MAJOR} "." ${cuPoly_VERSION_MINOR} "." ${cuPoly_VERSION_PATCH} " (" ${cuPoly_VERSION_TWEAK} ")")
MESSAGE(STATUS "cuPOLY found: " ${CUPOLY_LIBRARIES} " " ${CUPOLY_VERSION})

include(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(CUPOLY DEFAULT_MSG CUPOLY_INCLUDE_DIR CUPOLY_LIBRARIES)