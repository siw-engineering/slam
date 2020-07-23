# slam
# Elastic Fusion_Surfel Mesh 

![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)

This repo focusing  on integration of live feed Realsense with Elastic fusion odometry and Surfel based reconstrucion.
  - RealSense [Installation](https://github.com/IntelRealSense/librealsense/blob/master/doc/distribution_linux.md) 
  - [Elastic Fusion](https://github.com/mp3guy/ElasticFusion) 
  - [Surfel Meshing](https://github.com/puzzlepaint/surfelmeshing)

# Installation

  - Open the Elastic Fusion Folder, Run the build.sh file.
    ```sh
    $ ./build.sh
  - After installation follow these,
    ```sh
    $ mkdir build & cd build
    $ cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo -DCMAKE_CUDA_FLAGS="-arch=sm_61" ..
  - goto ./build/applications/surfel_meshing_CMakeFiles/SurfelMeshing.dir/
  - open link.txt add the following lines after anaconda3/lib:
     ```sh
    ../../../ElasticFusion/Core/build:../../../ElasticFusion/deps/Pangolin/build/src -lz ../../../ElasticFusion/deps/Pangolin/build/src/libpangolin.so 
  - Add the below one at the end of the link.txt
     ```sh
     -ldc1394 /usr/local/lib/librealsense2.so

  - After that Finall make .
    ```sh 
     $make -j SurfelMeshing
# Note :
 Code has been completed till odometry (live realsense feed), reconstruction still under progress. It will load given dataset and realsense video feed simultaneously (debug)

### Run
  ```sh
  ./applications/surfel_meshing/SurfelMeshing ../rgbd_dataset_freiburg1_floor groundtruth.txt --restrict_fps_to 30 --follow_input_camera false --outlier_filtering_required_inliers 0 --max_surfel_confidence 1 --max_depth 6


