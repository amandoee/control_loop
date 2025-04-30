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
        self.cumulative_delta = [0,0,0]
        self.odom_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.odom_callback, 1)

        self.driftpose = None
        self.driftodom = None
        self.has_drifted = False

        self.drift_detection_timer = self.create_timer(0.05, self.drift_detection)


        
    def scan_callback(self, msg):
            # Store the latest lidar scan
            self.last_scan = msg


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



                if self.has_drifted:
                    self._logger.info("running on odom")
                    self.current_x = self.current_x + self.cumulative_delta[0]
                    self.current_y = self.current_y + self.cumulative_delta[1]
                    self.current_yaw = self.current_yaw + self.cumulative_delta[2]
                    

                    flipped_y_origin = (800 - abs(self.origin[1]/self.map_resolution))*2-self.origin[1]/self.map_resolution
                    
                    #set axis as 0-1600
                    #plt.axis([0, 1600, 0, 1600])
                    #plt.scatter(-self.origin[0] / self.map_resolution+self.current_x/self.map_resolution, flipped_y_origin-self.current_y/self.map_resolution, c='red', s=100)
                    #plt.imshow(self.coordinates_with_data, cmap='gray', extent=[-self.origin[0], 1600-self.origin[0], -self.origin[1], 1600-self.origin[1]], alpha=0.5)
                    #plt.show()

                    y_estimate = -self.origin[0] / self.map_resolution+self.current_x/self.map_resolution
                    x_estimate = flipped_y_origin-self.current_y/self.map_resolution

                    self.xRange = [int(x_estimate - self.rangesize), int(x_estimate + self.rangesize)]
                    self.yRange = [int(y_estimate - self.rangesize), int(y_estimate + self.rangesize)]
                    


                else:
                    self.xRange = [int(best_xy[0][0] - self.rangesize), int(best_xy[0][0] + self.rangesize)]
                    self.yRange = [int(best_xy[1][0] - self.rangesize), int(best_xy[1][0] + self.rangesize)]
                    
                    # Update current pose based on the processed scan
                    self.current_x = x_coord#(x_coord + self.current_x)/2
                    self.current_y = y_coord#(y_coord + self.current_y)/2
                    self.current_yaw = np.deg2rad(float(best_angle.value) - 90)



                self.cumulative_delta = [0, 0, 0]

                self.initizalized = True
                # Publish a marker circle around the updated current pose
                self.publish_circle_marker(self.current_x, self.current_y)
        
    def odom_callback(self, msg):
         
        #Get yaw from odometry
        quat = msg.pose.pose.orientation
        _, _, msg_yaw = euler_from_quaternion([quat.x, quat.y, quat.z, quat.w])


        if self.old_odompose is not None:

            self.delta_odompose = [
                msg.pose.pose.position.x - self.old_odompose[0],
                msg.pose.pose.position.y - self.old_odompose[1],
                msg_yaw - self.old_odompose[2]
            ]
 
            self.cumulative_delta = [self.delta_odompose[0]+self.cumulative_delta[0], self.delta_odompose[1]+self.cumulative_delta[1], self.delta_odompose[2]+self.cumulative_delta[2]]
            
            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg_yaw
            ]

        elif self.initizalized:
            self.old_odompose = [
                msg.pose.pose.position.x,
                msg.pose.pose.position.y,
                msg_yaw
            ]


    def drift_detection(self):
        #Calculate diffrence between current_x and current_y and the cumulative delta

        if self.driftpose is not None and self.driftodom is not None:


            diff_x = self.driftpose[0] - self.current_x
            diff_y = self.driftpose[1] - self.current_y

            diff_odom_x = self.driftodom[0] - self.old_odompose[0]
            diff_odom_y = self.driftodom[1] - self.old_odompose[1]

            #Check if differences are similar
            percentage_diff_x = abs(abs(diff_x-diff_odom_x) / ((diff_x+diff_odom_x)/2))
            percentage_diff_y = abs(abs(diff_y-diff_odom_y) / ((diff_y+diff_odom_y)/2))
            #If the difference is greater than 10%, we have a drift


            self._logger.info(f"Drift detection: {percentage_diff_x}, {percentage_diff_y}")
            if (percentage_diff_x > 0.05 or percentage_diff_y > 0.05) and self.speed >= 5.:
                self._logger.info("Drift detected")
                self.has_drifted = True
            else:
                self.has_drifted = False
                

                
                
        elif self.old_odompose is not None and self.initizalized:
            self.driftpose = [self.current_x, self.current_y]
            self.driftodom = [self.old_odompose[0], self.old_odompose[1]]



def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
