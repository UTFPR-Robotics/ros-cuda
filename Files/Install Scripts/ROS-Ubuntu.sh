#!/bin/bash
# Script made to go along with the tutorial chapter GPU and ROS for Springer ROS Book 2018 (Vol 3)
# Created by Nicolas Dalmedico
# This script will install ROS Kinetic on your Ubuntu 16.04 PC
# Downloaded from: https://github.com/air-lasca/ros-cuda
# More information about this script can be found inside the chapter.
# If any problem is found report at: https://github.com/air-lasca/ros-cuda/issues

# Allow packages from packages.ros.org
sudo sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list'
# Setup keys
sudo apt-key adv --keyserver hkp://ha.pool.sks-keyservers.net:80 --recv-key 421C365BD9FF1F717815A3895523BAEEB01FA116
# Update package lists
sudo apt-get update
# Install ROS desktop
sudo apt-get install ros-kinetic-desktop-full -y
# Initialize rosdep and update
sudo rosdep init
rosdep update
# Setup environment variables
echo "source /opt/ros/kinetic/setup.bash" >> ~/.bashrc
source ~/.bashrc
# Create the workspace and init
mkdir -p ~/catkin_ws/src
cd ~/catkin_ws/
catkin_make
# Allow ROS to find the packages
echo "source ~/catkin_ws/devel/setup.bash" >> ~/.bashrc
source ~/.bashrc

