cmake_minimum_required(VERSION 3.1.0)
project(slam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
find_package(cv_bridge_1_13 REQUIRED)


find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg image_transport)  
include_directories(${catkin_INCLUDE_DIRS}) 
catkin_package()

find_package(OpenCV REQUIRED)
find_package(CUDA REQUIRED)

# set(OpenCV_DIR "/home/developer/co-fusion/deps/opencv-3.1.0/build")

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


# Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})

file(GLOB srcs *.cpp)


add_executable(${PROJECT_NAME}
               ${srcs}
)
