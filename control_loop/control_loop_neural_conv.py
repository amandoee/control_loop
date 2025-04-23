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
from control_loop.lidar_NN_conv import LocalizationNet
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
import csv
import math
import os
import json
import datetime





class AckermannLineFollower(AckermannLineConvParent):
    def __init__(self):
        AckermannLineConvParent.__init__(self)
        self.neuralnet = LocalizationNet()
        self.neuralnet.load_state_dict(torch.load('localization_model_conv.pth'))
        self.neuralnet.eval()
        #self.neuralnet = LocalizationNet()
        #self.neuralnet.load_state_dict(torch.load('localization_model.pth'))
        #self.neuralnet.eval()


    def estimate_pose_neural(self, msg):
        
        #Given a scan, estimate the next pose based on the change
        if self.old_scan is not None:

            #Calculate the change in the scan
            ranges = np.array(msg.ranges)
            ranges_old = np.array(self.old_scan.ranges)
            ranges_delta = ranges - ranges_old

            full_range = [0*i for i in range(1350)]
            for index, i in enumerate(range(round(self.current_yaw/0.0043633),round(1080+self.current_yaw/0.0043633))):
                if index >= 1080:
                    break
                full_range[i % 1350] = ranges_delta[index] 

            #Convert to 1350x1 tensor
            full_range_tensor = torch.tensor(full_range,dtype=torch.float32).view(1,1350)
            print(torch._shape_as_tensor(full_range_tensor))
            pred = self.neuralnet(full_range_tensor,torch.tensor([]))

            pred = pred.detach().numpy()
            print("Predicted: ",pred)
            
            x_delta_pred, y_delta_pred, yaw_delta_pred = pred[0][0],pred[0][1],pred[0][2]


            self.current_yaw += yaw_delta_pred
            self.current_x += x_delta_pred
            self.current_y += y_delta_pred

            print("current x: ",self.current_x)
            print("current y: ",self.current_y)
            print("current yaw: ",self.current_yaw)

        self.old_scan = msg
            #print



def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
