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
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import csv
import math
import os
import json
import datetime
import matplotlib.pyplot as plt
#import yaml loader
import yaml



class AckermannLineFollower(AckermannLineConvParent):
    def __init__(self):
        AckermannLineConvParent.__init__(self)


    

def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
