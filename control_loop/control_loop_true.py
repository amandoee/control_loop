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
from tf_transformations import euler_from_quaternion
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
import os
import json
import datetime
import matplotlib.pyplot as plt




class AckermannLineFollower(AckermannLineParent):
    def __init__(self):
        AckermannLineParent.__init__(self)
        self.true_pose = self.create_subscription(Odometry, 'ego_racecar/odom', self.set_pose, 1)
        self._logger.info("True pose subscriber initialized")
        self.max_speed = 3.

    def set_pose(self, msg):
        #Get the Pose2D from the message
        self.current_x = msg.pose.pose.position.x
        self.current_y = msg.pose.pose.position.y
        
        quat = msg.pose.pose.orientation
        _, _, self.current_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])

        self.initizalized = True



def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
