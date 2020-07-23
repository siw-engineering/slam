mkdir deps &> /dev/null
cd deps

#Add necessary extra repos
version=$(lsb_release -a 2>&1)


#Installing Pangolin
git clone https://github.com/stevenlovegrove/Pangolin.git
cd Pangolin
mkdir build
cd build
cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON
make -j8
cd ../..

#Up to date OpenNI2
git clone https://github.com/occipital/OpenNI2.git
cd OpenNI2
make -j8
cd ..

#Actually build ElasticFusion
cd ../Core
mkdir build
cd build
cmake ../src
make -j8
cd ../../GPUTest
mkdir build
cd build
cmake ../src
make -j8
cd ../../GUI
mkdir build
cd build
cmake ../src
make -j8
