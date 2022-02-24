cmake_minimum_required(VERSION 3.1.0)
project(slam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
SET(APP "apps/elastic_fusion.cpp" CACHE STRING "program to run")
SET(RS 0 CACHE INT "Some user-specified option")
SET(ROS 0 CACHE INT "Some user-specified option")

set(libconfig_INCLUDE_DIRS "/home/developer/deps/libconfig-1.7.3/lib" CACHE STRING "libconfig include dir")
if( "${MultipleModel}" STREQUAL "ON" )
  add_definitions(-DenableMultipleModel)
endif()

#find_package(cv_bridge_1_13 REQUIRED)
#set(OpenCV_DIR "/home/developer/co-fusion/deps/opencv-3.1.0/build")
#set(cv_bridge_DIR "/home/developer/packages/vision_opencv/cv_bridge/install/share/cv_bridge/cmake")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs sensor_msgs nav_msgs geometry_msgs genmsg cv_bridge image_transport tf2 tf2_geometry_msgs tf2_sensor_msgs tf2_ros tf2_eigen)
include_directories(${catkin_INCLUDE_DIRS}) 
catkin_package()


find_package(OpenCV 3.2 REQUIRED)
find_package(CUDA REQUIRED)

#realsense
if(${RS})
  find_package(RealSense2 REQUIRED)
endif()

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
if(${RS})
  include_directories(${REALSENSE2_INCLUDE_DIR})
endif()


# Warnings / Errors
add_compile_options(-std=c++14 -Wall -Wno-unused-function -Wno-unused-variable
  -Wno-unused-but-set-variable -Wno-unused-but-set-variable -Wno-write-strings
  -Wno-deprecated -Wno-deprecated-declarations -Wno-unknown-pragmas)

# Don't follow symlinks when FILE GLOB_RECURSE (and don't warn)
cmake_policy(SET CMP0009 NEW)

set(CUDA_HOST_COMPILER "/usr/bin/g++")
set(BUILD_TESTS OFF CACHE BOOL "Build unit tests")


# boost
find_package(Boost 1.58.0 REQUIRED)
message("Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})

# ncnn
if( "${MultipleModel}" STREQUAL "ON" )
#   set(ncnn_DIRS "/home/developer/works/interop/slam/src/third_party/ncnn")
  set(ncnn_config "/home/developer/deps/ncnn/build/install/lib/cmake/ncnn/")
#   add_subdirectory(${ncnn_DIRS})
  find_package(ncnn REQUIRED PATHS /home/developer/deps/ncnn/build/install/lib/cmake/ncnn/)
endif()

# Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

#pangolin
find_package(Pangolin 0.1 REQUIRED)
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})

#libconfig
set(libconfig_LIBRARY_DIRS "/usr/local/lib/")
find_package(libconfig REQUIRED)
message("libconfig_INCLUDE_DIRS: ${libconfig_INCLUDE_DIRS}")
include_directories(${libconfig_INCLUDE_DIRS})
link_directories(${libconfig_LIBRARY_DIRS})


#SuiteSparse
find_package(SuiteSparse REQUIRED)
include_directories(${SUITESPARSE_INCLUDE_DIRS})

# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})


file(GLOB gl gl/*.cpp)
file(GLOB inputs inputs/*.cpp)
file(GLOB inputs_ros inputs/ros/*.cpp)

file(GLOB model model/*.cpp)
file(GLOB odom odom/*.cpp)
file(GLOB lc lc/*.cpp)
file(GLOB tools tools/*.cpp)
file(GLOB ui ui/*.cpp)
if( "${MultipleModel}" STREQUAL "ON" )
  file(GLOB seg segmentation/*.cpp)
  file(GLOB seg_cuda segmentation/*.cu)
endif()
# file(GLOB cuda cuda/*.cu)
file(GLOB odom_cuda odom/*.cu)
file(GLOB of_cuda of/*.cu)
file(GLOB cuda_containers cuda/containers/*.cpp)
file(GLOB sf sf/*.cpp)


set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda_objs ${cuda})

CUDA_COMPILE(odom_cuda_objs ${odom_cuda}
                            ${seg_cuda})
CUDA_COMPILE(of_cuda_objs ${of_cuda})



add_executable(${PROJECT_NAME}
               ${ui}
               ${APP}
               ${gl}
               ${inputs}
               ${inputs_ros}
               ${model}
      			   ${odom}
               ${seg}
               ${seg_cuda}
               ${lc}
      			   ${tools}
               ${cuda}
               ${cuda_objs}
               ${odom_cuda}
               ${odom_cuda_objs}
               ${of_cuda_objs}
               ${cuda_containers}
               ${sf}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${BOOST_INCLUDE_DIRS}
    ${Pangolin_INCLUDE_DIRS}
    ${libconfig_LIBRARY_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    ${Eigen_LIBRARIES}
    ${cv_bridge_LIBRARIES}
    ${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    ${Pangolin_LIBRARIES}
    ${libconfig_LIBRARIES}
    ${SUITESPARSE_LIBRARIES}
    ${REALSENSE2_LIBRARY}
#     -ltbb
    boost_filesystem
    boost_system
    pthread
    config++
	)

if( "${MultipleModel}" STREQUAL "ON" )
target_link_libraries(${PROJECT_NAME}
                ncnn
                -lcuda)
endif()

if(${RS})
target_link_libraries(${PROJECT_NAME}
    realsense2
    )
endif()

