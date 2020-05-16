cmake_minimum_required(VERSION 2.6.0)

project(exp)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}")

# Pangolin
find_package(Pangolin 0.1 REQUIRED)
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})

# Eigen3
find_package(PkgConfig)
pkg_search_module(Eigen3 REQUIRED eigen3)
include_directories(${Eigen3_INCLUDE_DIRS})

include_directories(${EIGEN_INCLUDE_DIRS})
include_directories(${Pangolin_INCLUDE_DIRS})

message("Include-directories: ")
message("EIGEN_INCLUDE_DIRS: ${EIGEN_INCLUDE_DIRS}")
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
message("CMAKE_CURRENT_SOURCE_DIR/../Core: ${${CMAKE_CURRENT_SOURCE_DIR}/../Core}")

file(GLOB srcs pangolin.cpp)

set(SHADER_INCLUDE_DIRS "/home/christie/projects/work/siw/slam/slam_dev/cfs/Core/Shaders/")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -Wall -g -ljpeg -msse -msse2 -msse3 -I/usr/include/eigen3/ -fPIC -I/usr/include/python3.7m")


add_executable(exp
               ${srcs}
)

target_link_libraries(exp
                      ${Pangolin_LIBRARIES}
)


target_include_directories(exp PUBLIC
                            ${Pangolin_INCLUDE_DIRS}
                            ${SHADER_INCLUDE_DIRS}
)

