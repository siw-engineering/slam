#!/bin/bash
IMG="$1"
SLAM_REPO_PATH="$2"
SLAM_DATASET_PATH="$3"
WL_INTERFACE="$4"

if [ -z "$1" ]
then
      echo "Docker image cannot be empty"
      exit
fi

if [ -z "$2" ]
then
      echo "SLAM_REPO_PATH cannot be empty"
      exit
fi

if [ -z "$3" ]
then
      echo "SLAM_DATASET_PATH cannot be empty."
      exit

fi

if [ -z "$4" ]
then
      echo "WL_INTERFACE cannot be empty. (Run ifconfig and copy the wl interace id. (eg:wlxd0356714af3b)"
      exit

fi

IP=$(ifconfig "$WL_INTERFACE" | grep -Eo 'inet (addr:)?([0-9]*\.){3}[0-9]*' | grep -Eo '([0-9]*\.){3}[0-9]*' | grep -v '127.0.0.1')
PORT=11311
# Make sure processes in the container can connect to the x server
# Necessary so gazebo can create a context for OpenGL rendering (even headless)
XAUTH=/tmp/.docker.xauth
if [ ! -f $XAUTH ]
then
    xauth_list=$(xauth nlist $DISPLAY)
    xauth_list=$(sed -e 's/^..../ffff/' <<< "$xauth_list")
    if [ ! -z "$xauth_list" ]
    then
        echo "$xauth_list" | xauth -f $XAUTH nmerge -
    else
        touch $XAUTH
    fi
    chmod a+r $XAUTH
fi


DOCKER_OPTS="$DOCKER_OPTS --gpus all"

  # -e ROS_HOSTNAME=127.0.0.1 \
  # -e ROS_MASTER_URI=http://127.0.0.1:11311/ \
  # -v "/home/christie/projects/work/siw/subt/rosbag/:"$ROSBAG_MOUNT_DIR \
  	
docker run -it \
  -e DISPLAY \
  -e QT_X11_NO_MITSHM=1 \
  -e XAUTHORITY=$XAUTH \
  -e ROS_MASTER_URI=http://$IP:$PORT/ \
  -e ROS_HOSTNAME=$IP \
  -v "$XAUTH:$XAUTH" \
  -v "/tmp/.X11-unix:/tmp/.X11-unix" \
  -v "/dev/input:/dev/input" \
  -v "/dev/HID-SENSOR-2000e1.4.auto:/dev/HID-SENSOR-2000e1.4.auto" \
  -v "$SLAM_REPO_PATH:/home/developer/slam" \
  -v "$SLAM_DATASET_PATH:/home/developer/datasets" \
  --network host \
  --rm \
  --privileged \
  --security-opt seccomp=unconfined \
  $DOCKER_OPTS \
  $IMG \
  bash
