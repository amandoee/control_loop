import ctypes
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from control_loop.parent import AckermannLineParent
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from control_loop.lidar_NN import LocalizationNet
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import csv
import math
from geometry_msgs.msg import Pose2D
import os
import json
import datetime
import matplotlib.pyplot as plt
from tf_transformations import euler_from_quaternion


class AckermannLineFollower(AckermannLineParent):
    def __init__(self):
        AckermannLineParent.__init__(self)
        self.robot_pose_sub = self.create_subscription(Pose2D, 'robot_pose', self.slam_set_pose, 1)
        #self.odom_subber = self.create_subscription(Odometry, 'ego_racecar/odom', self.slam_set_pose_odom, 1)
        self.max_speed=5.


    def slam_set_pose(self, msg):
        #Get the Pose2D from the message
        if self.initizalized:
            self.current_x = msg.x
            self.current_y = msg.y
            self.current_yaw = msg.theta
            print("current x: ", self.current_x)
            print("current y: ", self.current_y)
            print("current yaw: ", self.current_yaw)
        self.initizalized = True

    


    #For Mapping with SLAM.
    # def slam_set_pose_odom(self,msg):
    #     #Get the Odometry from the message
    #     if self.initizalized:
    #         self.current_x = msg.pose.pose.position.x
    #         self.current_y = msg.pose.pose.position.y
    #         _, _, self.current_yaw = euler_from_quaternion([msg.pose.pose.orientation.x,
    #                                                      msg.pose.pose.orientation.y,
    #                                                      msg.pose.pose.orientation.z,
    #                                                      msg.pose.pose.orientation.w])
            
            
    #         print("current x: ", self.current_x)
    #         print("current y: ", self.current_y)
    #         print("current yaw: ", self.current_yaw)
    #     self.initizalized = True


def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
