# Ubuntu 18.04 with nvidia-docker2 beta opengl support
FROM nvidia/cudagl:10.2-devel-ubuntu18.04


# Tools I find useful during development
RUN apt-get update -qq \
 && apt-get install -y -qq \
        build-essential \
        bwm-ng \
        cmake \
        cppcheck \
        gdb \
        git \
        libbluetooth-dev \
        libcwiid-dev \
        libgoogle-glog-dev \
        libspnav-dev \
        libusb-dev \
        lsb-release \
        python3-dbg \
        python3-empy \
        python3-numpy \
        python3-setuptools \
        python3-pip \
        python3-venv \
        ruby2.5 \
        ruby2.5-dev \
        software-properties-common \
        sudo \
        vim \
        wget \
        net-tools \
        iputils-ping \
 && apt-get clean -qq

# Add a user with the same user_id as the user outside the container
# Requires a docker build argument `user_id`
ARG user_id
ENV USERNAME developer
RUN useradd -U --uid ${user_id} -ms /bin/bash $USERNAME \
 && echo "$USERNAME:$USERNAME" | chpasswd \
 && adduser $USERNAME sudo \
 && echo "$USERNAME ALL=NOPASSWD: ALL" >> /etc/sudoers.d/$USERNAME

# Commands below run as the developer user
USER $USERNAME

# Make a couple folders for organizing docker volumes
RUN mkdir ~/workspaces ~/other 
RUN mkdir ~/slam ~/datasets
RUN mkdir ~/.ignition



# When running a container start in the developer's home folder
WORKDIR /home/$USERNAME

RUN export DEBIAN_FRONTEND=noninteractive \
 && sudo apt-get update -qq \
 && sudo -E apt-get install -y -qq \
    tzdata \
 && sudo ln -fs /usr/share/zoneinfo/America/Los_Angeles /etc/localtime \
 && sudo dpkg-reconfigure --frontend noninteractive tzdata \
 && sudo apt-get clean -qq

# install ROS and required packages
RUN sudo /bin/sh -c 'echo "deb [trusted=yes] http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' \
 && sudo apt-get update -qq \
 && sudo apt-get install -y -qq \
    python-catkin-tools \
    python-rosdep \
    python-rosinstall \
    ros-melodic-desktop \
    ros-melodic-joystick-drivers \
    ros-melodic-pointcloud-to-laserscan \
    ros-melodic-robot-localization \
    ros-melodic-spacenav-node \
    ros-melodic-tf2-sensor-msgs \
    ros-melodic-tf2-eigen \
    ros-melodic-twist-mux \
    ros-melodic-rviz-imu-plugin \
    ros-melodic-rotors-control \
 && sudo rosdep init \
 && sudo apt-get clean -qq

RUN rosdep update

# sdformat8-sdf conflicts with sdformat-sdf installed from gazebo
# so we need to workaround this using a force overwrite
# Do this before installing ign-gazebo
RUN sudo /bin/sh -c 'echo "deb [trusted=yes] http://packages.osrfoundation.org/gazebo/ubuntu-stable `lsb_release -cs` main" > /etc/apt/sources.list.d/gazebo-stable.list' \
 && sudo /bin/sh -c 'wget http://packages.osrfoundation.org/gazebo.key -O - | sudo apt-key add -' \
 && sudo /bin/sh -c 'apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv-key C1CF6E31E6BADE8868B172B4F42ED6FBAB17C654'

# install ign-blueprint
RUN sudo apt-get update -qq \
 && sudo apt-get install -y -qq \
    ignition-blueprint \
 && sudo apt-get clean -qq

# install the ros to ign bridge
RUN sudo apt-get update -qq \
 && sudo apt-get install -y -qq \
    ros-melodic-ros-ign \
    ros-melodic-teleop-twist-keyboard \
 && sudo apt-get clean -qq

RUN sudo apt-get install -y -qq \
    apt-transport-https \
    ca-certificates \
&& sudo apt-get clean -qq


RUN curl -fsSL https://download.sublimetext.com/sublimehq-pub.gpg | sudo apt-key add - \
 && sudo add-apt-repository "deb https://download.sublimetext.com/ apt/stable/" \
 && sudo apt update \
 && sudo apt install sublime-text 

RUN mkdir -p ~/subt_ws/src/subt \
 && cd ~/subt_ws/src \
 && git clone https://github.com/osrf/subt

WORKDIR /home/$USERNAME/

RUN /bin/bash -c 'source /opt/ros/melodic/setup.bash'
RUN /bin/sh -c 'echo ". /opt/ros/melodic/setup.bash" >> ~/.bashrc'


RUN sudo apt-get install -y cmake-qt-gui git build-essential libusb-1.0-0-dev libudev-dev freeglut3-dev libglew-dev libsuitesparse-dev libeigen3-dev zlib1g-dev libjpeg-dev

RUN mkdir ~/deps \
&& cd ~/deps \
&& git clone --single-branch --branch v0.6 https://github.com/stevenlovegrove/Pangolin.git \
&& cd Pangolin \
&& mkdir build \
&& cd build \
&& cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON \
&& make -j8 


RUN cd ~/deps/ \
&& wget https://hyperrealm.github.io/libconfig/dist/libconfig-1.7.3.tar.gz \
&& tar -xvf libconfig-1.7.3.tar.gz \
&& cd libconfig-1.7.3 \
&& ./configure \
&& sudo make install \
