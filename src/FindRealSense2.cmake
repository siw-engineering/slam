set(REALSENSE2_ROOT "/usr/local" CACHE PATH "Root directory of libREALSENSE2")

find_path(REALSENSE2_INCLUDE_DIR NAMES librealsense librealsense2 HINTS "${REALSENSE2_ROOT}/include")
find_library(REALSENSE2_LIBRARY NAMES realsense realsense2 HINTS "${REALSENSE2_ROOT}/bin/x64" "${REALSENSE2_ROOT}/lib")

find_package_handle_standard_args(REALSENSE2 DEFAULT_MSG REALSENSE2_LIBRARY REALSENSE2_INCLUDE_DIR)

mark_as_advanced(REALSENSE2_LIBRARY REALSENSE2_INCLUDE_DIR)