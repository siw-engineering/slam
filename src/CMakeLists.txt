cmake_minimum_required(VERSION 3.1.0)
project(slam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
find_package(cv_bridge_1_13 REQUIRED)

set(OpenCV_DIR "/home/developer/co-fusion/deps/opencv-3.1.0/build")
set(cv_bridge_DIR "/home/developer/packages/vision_opencv/cv_bridge/install/share/cv_bridge/cmake")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg image_transport)  
include_directories(${catkin_INCLUDE_DIRS}) 
catkin_package()

find_package(OpenCV 3.1 REQUIRED)
find_package(CUDA REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})



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
file(GLOB inputs_ros inputs/ros/*.cpp)


add_executable(${PROJECT_NAME}
               ${srcs}
               ${inputs_ros}
)

target_include_directories(${PROJECT_NAME} PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${CUDA_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${BOOST_INCLUDE_DIRS}
)

target_link_libraries(${PROJECT_NAME}
    ${Eigen_LIBRARIES}
    ${cv_bridge_LIBRARIES}
	${catkin_LIBRARIES}
    ${CUDA_LIBRARIES}
    ${OpenCV_LIBRARIES}
    boost_filesystem
    boost_system
    pthread
	)

