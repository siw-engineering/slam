#!/bin/bash

mkdir ~/deps &> /dev/null

cd ~/deps
git clone --single-branch --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j8

cd ~/deps
wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.3.tar.gz
tar -xvf libconfig-1.7.3.tar.gz
cd libconfig-1.7.3
./configure
sudo make install


#install realsense
sudo apt-key adv --keyserver keys.gnupg.net --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE || sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key F6E65AC044F831AC80A06380C8B3A55A6F3EFCDE
sudo add-apt-repository "deb https://librealsense.intel.com/Debian/apt-repo $(lsb_release -cs) main" -u
sudo apt-get install librealsense2-dkms librealsense2-utils librealsense2-dev librealsense2-dbg -y
