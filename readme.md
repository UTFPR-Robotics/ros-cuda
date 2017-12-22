# GPU and ROS: The use of general parallel processing architecture for robot perception

This repository is used to store ROS packages and files that are used as examples for the tutorial chapter GPU and ROS in the Springer ROS Book 2018 (Vol 3).

## Getting Started

If you do not have the ROS Book 2018 chapter at your disposal, you can also use the sample packages by doing the following

### Prerequisites

To use the packages, you will need any version of ROS above Indigo and you also need CUDA 5.0 or above installed on your Ubuntu.

### ROS Packages

To use the ROS packages, simply drop them inside your ROS workspace and use 'catkin_make' to compile. Every example has a launch file to allow easy use as follows:

```
roslaunch roscuda_vectoradd roscuda_vectoradd.launch
```

This should be enough to run every example. Some of the packages have ROSbag files (inside the respective folder) that allow you to visualize the result if you do not have the sensors or image streams required by the packages. To run a ROSbag file, one can type:

```
rosbag play -l path/to/the/bag.bag
```

### Files

The files inside the 'files' folder are scripts or replacement files needed for ROS installation in the Nvidia Tegra boards. They're only necessary for those using these boards. Read the chapter for more information.

## Authors

* **Nicolas Dalmedico**
* **Marco Antônio Simões Teixeira**

See also the list of [contributors](https://github.com/air-lasca/ros-cuda/contributors) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details
