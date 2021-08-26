#!/bin/bash

mkdir deps &> /dev/null
cd deps

git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j8
cd ../..

wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.3.tar.gz
tar -xvf libconfig-1.7.3.tar.gz
cd libconfig-1.7.3
./configure
sudo make install
cd ../

catkin_make