# slam
SLAM system

## Build
Build system : catkin_make <br>
Assuming Opencv and CUDA is installed, run build.sh. <br>
eg:
```
./build.sh
catkin_make -DAPP=apps/elastic_fusion_file.cpp
```

## Dependencies 
* OpenCV 3.2
* CUDA
* [Pangolin](https://github.com/stevenlovegrove/Pangolin)
* [libconfig](https://github.com/hyperrealm/libconfig.git)

## Benchmark
Benchmark is done on the iclnuim [living room dataset](http://www.doc.ic.ac.uk/~ahanda/living_room_traj0_frei_png.tar.gz)
Metrics | ATE | RPE | PC Alignment score
--- | --- | --- | --- |
Score | 0.018483 | 0.0231662926318 | 0.00750245 | 
