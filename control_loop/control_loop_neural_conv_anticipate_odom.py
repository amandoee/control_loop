import ctypes
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
from control_loop.parent import AckermannLineConvParent, AckermannLineParent
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
        self.cumulative_delta = [0,0]
        self.delta_since_lost = [0,0]

        self.last_reliable_position = None

        self.odom_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 1)
        
    def scan_callback(self, msg):
            # Store the latest lidar scan
            self.last_scan = msg

            print("Scan callback for verification that override is working")

            result = np.zeros((self.map_size, self.map_size), dtype=np.int32)
            best_sum = ctypes.c_int(-1)
            best_angle = ctypes.c_int(0)

            # Convert lidar scan to numpy array of x, y coordinates
            if self.last_scan is not None:
                ranges = np.array(self.last_scan.ranges)
                angles = np.linspace(self.last_scan.angle_min, self.last_scan.angle_max, len(ranges))
                x = ranges * np.cos(angles)
                y = ranges * np.sin(angles)

                x = np.round(x / self.map_resolution)
                y = np.round(y / self.map_resolution)
                valid_indices = ranges <= 29.8
                x = x[valid_indices]
                y = y[valid_indices]

                xy_lidar_local_lowrest = np.array(list(set(zip(x, y))))
                x_lidar_local = np.ascontiguousarray(xy_lidar_local_lowrest[:, 0][::4])
                y_lidar_local = np.ascontiguousarray(xy_lidar_local_lowrest[:, 1][::4])

                self.lib.convolve_lidar_scan_c_coarse_fine(
                    x_lidar_local,
                    y_lidar_local,
                    len(x_lidar_local),
                    self.coordinates_with_data.ravel(),
                    int(self.xRange[0]),
                    int(self.xRange[1]),
                    int(self.yRange[0]),
                    int(self.yRange[1]),
                    np.rad2deg(self.current_yaw)+90.,
                    90,
                    result.ravel(),
                    ctypes.byref(best_sum),
                    ctypes.byref(best_angle)
                )

                #If best angle is more different than 90 degrees, it is probably wrong, so we ignore it
                #print("Best angle: ", (float(best_angle.value) - 90))
                #print("Current yaw: ", np.rad2deg(self.current_yaw))

                #TODO: Check if rounding is correct.
                best_xy = np.where(result == best_sum.value)
                if len(best_xy[0]) > 1:
                    best_xy = (np.array([np.mean(best_xy[0])]), np.array([np.mean(best_xy[1])]))
                y_coord = (((800 - abs(self.origin[1] / self.map_resolution)) * 2 - self.origin[1] / self.map_resolution) - best_xy[0][0]) * self.map_resolution
                x_coord = (best_xy[1][0] + self.origin[0] / self.map_resolution) * self.map_resolution

                print("Best xy: ", x_coord, y_coord)

                #In odom version, we set the new range as the odometry estimate of where we are

                #self.xRange = [int(best_xy[0][0]+self.cumulative_delta[1]/self.map_resolution - self.rangesize), int(best_xy[0][0]+self.cumulative_delta[1]/self.map_resolution + self.rangesize)]
                #self.yRange = [int(best_xy[1][0]+self.cumulative_delta[0]/self.map_resolution - self.rangesize), int(best_xy[1][0]+self.cumulative_delta[0]/self.map_resolution + self.rangesize)]
                


                #Calculate the difference between self_current.x,self_current.y and the best_xy
                distance_scan = math.sqrt((self.current_x - x_coord)**2 + (self.current_y - y_coord)**2)
                distance_odom = math.sqrt((self.delta_since_lost[0]+self.cumulative_delta[0])**2 + (self.delta_since_lost[1]+self.cumulative_delta[1])**2)

                #Convert to map coordinates


                #If the two distances are more than 10% apart, ignore the scan estimate
                try:
                    percentage_difference = abs(distance_odom - distance_scan) / ((distance_odom+ distance_scan)/2)
                except:
                    percentage_difference=0.
                
                print("Percentage difference: ", percentage_difference)

                if percentage_difference > 1 and self.initizalized:
                    self.current_x+=self.cumulative_delta[0]
                    self.current_y+=self.cumulative_delta[1]

                    self.delta_since_lost = [self.delta_since_lost[0]+self.cumulative_delta[0], self.delta_since_lost[1]+self.cumulative_delta[1]]

            
                    #Set the new ranges based on the current position
                    
                    self.xRange = [int(best_xy[0][0]+self.delta_since_lost[0]/self.map_resolution - self.rangesize), int(best_xy[0][0]+self.delta_since_lost[0]/self.map_resolution + self.rangesize)]
                    self.yRange = [int(best_xy[1][0]+self.delta_since_lost[1]/self.map_resolution - self.rangesize), int(best_xy[1][0]+self.delta_since_lost[1]/self.map_resolution + self.rangesize)]
                    
                    print("Ignoring scan estimate")
                else:

                    self.current_x = x_coord#(x_coord + self.current_x)/2
                    self.current_y = y_coord#(y_coord + self.current_y)/2
                    
                    self.delta_since_lost = [0,0]

                    print("Potentially fucked ")

                    print("---")
                    self.xRange = [int(best_xy[0][0] - self.rangesize), int(best_xy[0][0] + self.rangesize)]
                    self.yRange = [int(best_xy[1][0] - self.rangesize), int(best_xy[1][0] + self.rangesize)]


                print("deltas: ", self.cumulative_delta)
                self.cumulative_delta = [0, 0]


                print(self.xRange)
                print(self.yRange)

                # Update current pose based on the processed scan

                self.current_yaw = np.deg2rad(float(best_angle.value) - 90)
                self.initizalized = True
                # Publish a marker circle around the updated current pose
                self.publish_circle_marker(self.current_x, self.current_y)
        
    def odom_callback(self, msg):
         
        if self.old_odompose is not None:

            self.delta_odompose = [
                msg.pose.pose.position.x - self.old_odompose[0],
                msg.pose.pose.position.y - self.old_odompose[1],
            ]
 
            self.cumulative_delta = [self.cumulative_delta[0]+self.delta_odompose[0], self.cumulative_delta[1]+self.delta_odompose[1]]
            
            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ]

        else:
            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
            ]


def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
