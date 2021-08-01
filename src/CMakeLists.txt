cmake_minimum_required(VERSION 3.1.0)
project(slam)

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_CURRENT_SOURCE_DIR}/")
set(Pangolin_DIR "/home/developer/co-fusion/deps/Pangolin/build")

#find_package(cv_bridge_1_13 REQUIRED)

#set(OpenCV_DIR "/home/developer/co-fusion/deps/opencv-3.1.0/build")
#set(cv_bridge_DIR "/home/developer/packages/vision_opencv/cv_bridge/install/share/cv_bridge/cmake")

find_package(catkin REQUIRED COMPONENTS roscpp rospy std_msgs genmsg cv_bridge image_transport)  
include_directories(${catkin_INCLUDE_DIRS}) 

catkin_package()

find_package(OpenCV 3.2 REQUIRED)
find_package(CUDA REQUIRED)


# find_package(RealSense2 REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})
# include_directories(${REALSENSE2_INCLUDE_DIR})


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

#pangolin
find_package(Pangolin 0.1 REQUIRED)
message("Pangolin_INCLUDE_DIRS: ${Pangolin_INCLUDE_DIRS}")
include_directories(${Pangolin_INCLUDE_DIRS})

#libconfig
set(libconfig_INCLUDE_DIRS "/home/developer/packages/libconfig-1.7.3/lib")
set(libconfig_LIBRARY_DIRS "/usr/local/lib/")
find_package(libconfig REQUIRED)
message("libconfig_INCLUDE_DIRS: ${libconfig_INCLUDE_DIRS}")
include_directories(${libconfig_INCLUDE_DIRS})
link_directories(${libconfig_LIBRARY_DIRS})


# project-wide paths
include_directories(${PROJECT_SOURCE_DIR})


file(GLOB apps apps/*.cpp)
file(GLOB gl gl/*.cpp)
file(GLOB inputs inputs/*.cpp)
file(GLOB model model/*.cpp)
file(GLOB odom odom/*.cpp)
file(GLOB tools tools/*.cpp)
file(GLOB ui ui/*.cpp)
# file(GLOB cuda cuda/*.cu)
file(GLOB odom_cuda odom/*.cu)
file(GLOB cuda_containers cuda/containers/*.cpp)



set(CUDA_ARCH_PTX "" CACHE STRING "Specify 'virtual' PTX arch to build PTX intermediate code for. Example: 1.0 1.2 or 10 12")
include(CudaComputeTargetFlags.cmake)
APPEND_TARGET_ARCH_FLAGS()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-Xcompiler;-fPIC;-D_FORCE_INLINES;")
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} "--ftz=true;--prec-div=false;--prec-sqrt=false")
if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}  "-g")
endif(CMAKE_BUILD_TYPE MATCHES Debug)

CUDA_COMPILE(cuda_objs ${cuda})
CUDA_COMPILE(odom_cuda_objs ${odom_cuda})


add_executable(${PROJECT_NAME}
               ${ui}
               ${apps}
               ${gl}
               ${inputs}
               ${model}
      			   ${odom}
      			   ${tools}
               ${cuda}
               ${cuda_objs}
               ${odom_cuda}
               ${odom_cuda_objs}
               ${cuda_containers}
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

    # ${REALSENSE2_LIBRARY}
    boost_filesystem
    boost_system
    pthread
    config++
	)

