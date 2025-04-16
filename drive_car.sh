#!/bin/bash

source install/local_setup.bash
sudo colcon build
sudo systemctl stop safety_ros_launch.service
ros2 run control_loop control_loop
