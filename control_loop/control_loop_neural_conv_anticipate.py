import ctypes
import re
import time
from matplotlib import pyplot as plt
import numpy as np
import torch
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


origin = [-2.89, -4.06]


def read_pgm(filename, byteorder='>'):
    with open(filename, 'rb') as f:
        buffer = f.read()
    try:
        header, width, height, maxval = re.search(
            b"(^P5\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n])*"
            b"(\d+)\s(?:\s*#.*[\r\n]\s)*)", buffer).groups()
        
        width = int(width)
        height = int(height)
        maxval = int(maxval)

        # Convert to numpy array
        image = np.frombuffer(buffer, dtype='u1' if maxval < 256 else byteorder+'u2',
                                count=int(width)*int(height), offset=len(header)).reshape((int(height), int(width)))

        coordinates_with_data = np.array(image, dtype=np.int32)

        # If image values are greater than occupied_thresh, set to 1, else 0
        image = image.copy().astype(float)
        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i, j] >=0.45:
                    image[i, j] = 0
                else:
                    image[i, j] = 1
                    coordinates_with_data[i, j] = 1

        #plot it


        return image, coordinates_with_data

    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)


def angular_distance(a, b):
    diff = (b - a + 180) % 360 - 180
    return abs(diff)

class AckermannLineFollower(Node):
    def __init__(self):
        super().__init__('ackermann_line_follower')
        self.initizalized = False
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 1)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 1)

        self.pose_draw_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_draw_callback, 1)


        #self.estimate_neural = self.create_subscription(LaserScan, 'scan', self.estimate_pose_neural, 1)
        self.timer = self.create_timer(0.001, self.control_loop)
        self.rangesize = 10

        self.old_scan = None
        self.old_yaw = 1.57079633


        self.old_x = 0.0
        self.old_y = 0.0
        self.old_time = time.time()
        self.current_time = time.time()


        #self.neuralnet = LocalizationNet()
        #self.neuralnet.load_state_dict(torch.load('localization_model.pth'))
        #self.neuralnet.eval()
        
        #self.lib = ctypes.CDLL('/home/f1t/test_cuda/cuda_test')
        
        self.lib = ctypes.CDLL('/home/f1t/au_f1tenth_ws/control_loop/plotting/cuda.so')

        #self.lib = ctypes.CDLL('./refined180range.so')
        self.lib.convolve_lidar_scan_c_coarse_fine.argtypes = [
            np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
            np.ctypeslib.ndpointer(dtype=np.double, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_double,
            ctypes.c_int,
            np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS'),
            ctypes.POINTER(ctypes.c_int),
            ctypes.POINTER(ctypes.c_int)
        ]
        self.lib.convolve_lidar_scan_c_coarse_fine.restype = None
        self.map_size = 1600
        self.map_resolution = 0.05
        self.map, self.coordinates_with_data = read_pgm('./maps/expandedcircle1600.pgm')
        #self.xRange = [-origin[1] / self.map_resolution - 200, -origin[1] / self.map_resolution + 200]
        #self.yRange = [-origin[0] / self.map_resolution - 200, -origin[0] / self.map_resolution + 200]
        self.xRange=[700,900]
        self.yRange=[700,900]
        self.old_target_index = 0
        
        # Load centerline data from CSV
        self.centerline = []
        file_path = './centerline/custom.csv'
        try:
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for row in csv_reader:
                    if len(row) < 2:
                        continue
                    x, y = float(row[0].strip()), float(row[1].strip())
                    self.centerline.append((x, y))
        except Exception as e:
            self.get_logger().error(f"Failed to load centerline data: {e}")
        
        if not self.centerline:
            self.get_logger().error("Centerline data is empty!")
        else:
            self.get_logger().info(f"Loaded centerline with {len(self.centerline)} points")
        
        # Start following from the second point (first is the spawn point)
        self.target_index = 1

        # Initialize current state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = np.pi/2
        # Storage for the latest lidar scan
        self.last_scan = None

        # Setup CSV file for logging movement data
        # self.movement_data_file = "./movement_data0.csv"
        # if not os.path.exists(self.movement_data_file):
        #     with open(self.movement_data_file, 'w', newline='') as csv_file:
        #         writer = csv.writer(csv_file, delimiter=";")
        #         writer.writerow(["timestamp", "x", "y", "yaw", "lidar_scan"])

        # Publisher for visualization markers (circle)
        self.marker_pub = self.create_publisher(Marker, 'visualization_marker', 1)


    def pose_draw_callback(self, msg):
        # Draw the current pose of the car
        xdraw = msg.pose.pose.position.x
        ydraw = msg.pose.pose.position.y
        self.publish_circle_marker(xdraw, ydraw, radius=0.05)


    def publish_circle_marker(self, x, y, radius=10*0.05, num_points=50):
        marker = Marker()
        marker.header.frame_id = "map"  # Adjust if needed to match your RViz frame
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = "circle"
        marker.id = 0
        marker.type = Marker.LINE_STRIP  # Use a continuous line strip to form the circle
        marker.action = Marker.ADD
        marker.scale.x = 0.1  # Line width

        # Set marker color to red (RGBA)
        marker.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0)

        # Generate points for the circle
        for i in range(num_points + 1):  # +1 to close the circle
            angle = 2 * np.pi * i / num_points
            pt = Point()
            pt.x = x + radius * np.cos(angle)
            pt.y = y + radius * np.sin(angle)
            pt.z = 0.0
            marker.points.append(pt)


        self.marker_pub.publish(marker)



    def estimate_pose_neural(self, msg):
        # Given a scan, estimate the next pose based on the change
        if self.old_scan is not None:
            ranges = np.array(msg.ranges)
            ranges_old = np.array(self.old_scan.ranges)
            ranges_delta = ranges - ranges_old

            full_range = [0 * i for i in range(1350)]
            for index, i in enumerate(range(round(self.current_yaw / 0.0043633), round(1080 + self.current_yaw / 0.0043633))):
                if index >= 1080:
                    break
                full_range[i % 1350] = ranges_delta[index]

            full_range_tensor = torch.tensor(full_range, dtype=torch.float32).view(1, 1350)
            print(torch._shape_as_tensor(full_range_tensor))
            pred = self.neuralnet(full_range_tensor, torch.tensor([]))
            pred = pred.detach().numpy()
            print("Predicted: ", pred)
            
            x_delta_pred, y_delta_pred, yaw_delta_pred = pred[0][0], pred[0][1], pred[0][2]

            self.current_yaw += yaw_delta_pred
            self.current_x += x_delta_pred
            self.current_y += y_delta_pred

            print("current x: ", self.current_x)
            print("current y: ", self.current_y)
            print("current yaw: ", self.current_yaw)
            #self.initizalized = True
            self.publish_circle_marker(self.current_x, self.current_y,radius=self.rangesize*0.05)


        self.old_scan = msg



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






            best_xy = np.where(result == best_sum.value)
            if len(best_xy[0]) > 1:
                best_xy = (np.array([np.mean(best_xy[0])]), np.array([np.mean(best_xy[1])]))
            y_coord = (((800 - abs(origin[1] / self.map_resolution)) * 2 - origin[1] / self.map_resolution) - best_xy[0][0]) * self.map_resolution
            x_coord = (best_xy[1][0] + origin[0] / self.map_resolution) * self.map_resolution
            self.xRange = [int(best_xy[0][0] - self.rangesize), int(best_xy[0][0] + self.rangesize)]
            self.yRange = [int(best_xy[1][0] - self.rangesize), int(best_xy[1][0] + self.rangesize)]
            # Update current pose based on the processed scan
            self.current_x = x_coord
            self.current_y = y_coord
            self.current_yaw = np.deg2rad(float(best_angle.value) - 90)
            self.initizalized = True
            # Publish a marker circle around the updated current pose
            self.publish_circle_marker(self.current_x, self.current_y)

    def log_waypoint_data(self):
        timestamp = datetime.datetime.now().isoformat()
        lidar_data = json.dumps(list(self.last_scan.ranges)) if self.last_scan is not None else "None"        
        with open(self.movement_data_file, 'a', newline='') as csv_file:
            writer = csv.writer(csv_file, delimiter=";")
            writer.writerow([timestamp, self.current_x, self.current_y, self.current_yaw, lidar_data])
        
    def control_loop(self):

        target_x, target_y = self.centerline[self.target_index]
        dx = target_x - self.current_x
        dy = target_y - self.current_y
        distance = math.hypot(dx, dy)

        if distance < 0.3:
            #self.log_waypoint_data()
            self.target_index += 1
            if self.target_index >= len(self.centerline):
                self.target_index = 0
            target_x, target_y = self.centerline[self.target_index]
            dx = target_x - self.current_x
            dy = target_y - self.current_y

            self.old_x = self.current_x
            self.old_y = self.current_y
            self.old_yaw = self.current_yaw

        desired_yaw = math.atan2(dy, dx)
        error_yaw = desired_yaw - self.current_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))
        kp = 0.50  # Adjust gain as needed
        steering_angle = kp * error_yaw

        msg = AckermannDriveStamped()
        msg.drive.speed = 1.  # Constant speed; adjust as needed
        msg.drive.steering_angle = steering_angle

        print("x: ", self.current_x)
        print("y: ", self.current_y)
        print("yaw: ", self.current_yaw)

        if self.initizalized:
            self.publisher_.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = AckermannLineFollower()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
