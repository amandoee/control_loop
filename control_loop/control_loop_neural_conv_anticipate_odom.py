import ctypes
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from control_loop.parent import AckermannLineConvParent
import rclpy
from rclpy.node import Node
from ackermann_msgs.msg import AckermannDriveStamped
from control_loop.lidar_NN import LocalizationNet
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from tf_transformations import euler_from_quaternion as euler_from_quaternion
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import csv
import math
import os
import json
import datetime
import matplotlib.pyplot as plt



class AckermannLineFollower(AckermannLineConvParent):
    def __init__(self):
        AckermannLineConvParent.__init__(self)
        self.old_odompose = None
        self.delta_odompose = None
        self.odom_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 1)
        
        
    def odom_callback(self, msg):
        # Calculate the delta from msg odom and old_odompose
        _,_, new_odomyaw = euler_from_quaternion(
                [msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w]
            )
         
        if self.old_odompose is not None:

            self.delta_odompose = [
                msg.pose.pose.position.x - self.old_odompose[0],
                msg.pose.pose.position.y - self.old_odompose[1],
                new_odomyaw - self.old_odompose[2]
            ]
            self.current_x += self.delta_odompose[0]
            self.current_y += self.delta_odompose[1]
            self.current_yaw += self.delta_odompose[2]

            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                new_odomyaw
            ]
        else:
            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                new_odomyaw
            ]


def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
