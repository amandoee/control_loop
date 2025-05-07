from matplotlib import pyplot as plt
import yaml
from rclpy.node import Node
import ctypes
import time
from ackermann_msgs.msg import AckermannDriveStamped
import re
import numpy as np
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry
import csv
import math
from tf_transformations import euler_from_quaternion, quaternion_from_euler
from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point
import torch
import datetime
from visualization_msgs.msg import Marker
import json
from geometry_msgs.msg import PoseWithCovarianceStamped



class AckermannLineParent(Node):
    def __init__(self):
        super().__init__('ackermann_line_follower')
        self.initizalized = False
        self.publisher_ = self.create_publisher(AckermannDriveStamped, 'drive', 1)
        

        
        self.declare_parameter('map_path', '')
        self.map_path = self.get_parameter('map_path').get_parameter_value().string_value
        self.declare_parameter('csv_path', '')
        self.csv_path = self.get_parameter('csv_path').get_parameter_value().string_value
        self.get_map_info(self.map_path+".yaml")

        #self.pose_draw_sub = self.create_subscription(Odometry, 'ego_racecar/odom', self.pose_draw_callback, 1)
        self.speed=0.7
        self.lap = 0
        #self.estimate_neural = self.create_subscription(LaserScan, 'scan', self.estimate_pose_neural, 1)
        self.timer = self.create_timer(0.001, self.control_loop)
        self.logger = self.create_subscription(Odometry, 'ego_racecar/odom', self.log_waypoint_data, 1)
        self.rangesize = 15

        self.old_scan = None

        self.old_estimate_pose = None

        self.max_speed = 8.0
        self.map_size = 0
        self.jumped_past = False



        # Initialize current state variables
        self.current_x = 0.0
        self.current_y = 0.0
        self.current_yaw = 0.5

        self.current_time = time.time()

        self.previous_error_yaw = 0.0
        self.last_control_time = self.get_clock().now()


        
        # Load centerline data from CSV
        self.centerline = []
        self.target_index = 0
        file_path = self.csv_path+'.csv'
        try:
            with open(file_path, 'r') as csv_file:
                csv_reader = csv.reader(csv_file)
                for index, row in enumerate(csv_reader):
                    if len(row) < 2:
                        continue
                    x, y = float(row[0].strip()), float(row[1].strip())
                    #If distance to point is between 0.5 and 1.0, and y is larger than 0, let this be the first target
                    self.centerline.append((x, y))
        except Exception as e:
            self.get_logger().error(f"Failed to load centerline data: {e}")
        
        if not self.centerline:
            self.get_logger().error("Centerline data is empty!")
        else:
            self.get_logger().info(f"Loaded centerline with {len(self.centerline)} points")

            #Create a publisher for initialpsoe
            self.initial_pose_pub = self.create_publisher(PoseWithCovarianceStamped, 'initialpose', 10)
            #Create a message for the initialpose
            initial_pose_msg = PoseWithCovarianceStamped()
            initial_pose_msg.header.frame_id = "map"
            initial_pose_msg.header.stamp = self.get_clock().now().to_msg()
            #Set the initial pose to the first point in the centerline
            initial_pose_msg.pose.pose.position.x = self.centerline[0][0]
            initial_pose_msg.pose.pose.position.y = self.centerline[0][1]
            initial_pose_msg.pose.pose.position.z = 0.0
            #Set the covariance to 0.1
            initial_pose_msg.pose.covariance[0] = 0.1
            initial_pose_msg.pose.covariance[7] = 0.1
            initial_pose_msg.pose.covariance[14] = 0.1
            initial_pose_msg.pose.covariance[21] = 0.1
            initial_pose_msg.pose.covariance[28] = 0.1
            initial_pose_msg.pose.covariance[35] = 0.1
            #Set the orientation to the current yaw
            
            #Calculate the angle between the first two points in the centerline
            initial_yaw = 0.0
            if len(self.centerline) > 1:
                dx = self.centerline[1][0] - self.centerline[0][0]
                dy = self.centerline[1][1] - self.centerline[0][1]
                initial_yaw = math.atan2(dy, dx)




            q = quaternion_from_euler(0, 0, initial_yaw)
            initial_pose_msg.pose.pose.orientation.x = q[0]
            initial_pose_msg.pose.pose.orientation.y = q[1]
            initial_pose_msg.pose.pose.orientation.z = q[2]
            initial_pose_msg.pose.pose.orientation.w = q[3]

            #Publish the initialpose
            self.initial_pose_pub.publish(initial_pose_msg)
            self.get_logger().info(f"Published initial pose: {initial_pose_msg.pose.pose.position.x}, {initial_pose_msg.pose.pose.position.y}")

            self.get_logger().info(f"Initial pose yaw: {euler_from_quaternion([q[0], q[1], q[2], q[3]])[2]}")

            
            #reset odometry topic



        # Storage for the latest lidar scan
        self.last_scan = None


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
            #self.publish_circle_marker(self.current_x, self.current_y,radius=self.rangesize*0.05)


        self.old_scan = msg



    def log_waypoint_data(self,msg):
        #get odometry data
        odom_x = msg.pose.pose.position.x
        odom_y = msg.pose.pose.position.y
        odom_yaw = euler_from_quaternion(
            [msg.pose.pose.orientation.x,
            msg.pose.pose.orientation.y,
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w]
        )[2]
        #get the current time
        timestamp = datetime.datetime.now().isoformat()
        #Note the estimated current pose, as well as odometry's current pose
        data = {
            'timestamp': timestamp,
            'current_speed': self.speed,
            'current_x': self.current_x,
            'current_y': self.current_y,
            'current_yaw': self.current_yaw,
            'ground_truth_x': odom_x,
            'ground_truth_y': odom_y,
            'ground_truth_yaw': odom_yaw,
            'lap': self.lap,
        }
        # Write the data to a CSV file
        file_path = './logged_data'+'with_speed_'+str(self.max_speed)+'.csv'
        with open(file_path, 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=data.keys())
            # Write the header only if the file is empty
            if csv_file.tell() == 0:
                writer.writeheader()
            writer.writerow(data)
        
    


    def control_loop(self):

        current_time = self.get_clock().now()
        


        target_x, target_y = self.centerline[self.target_index]
        dx = target_x - self.current_x
        dy = target_y - self.current_y


        distance = math.hypot(dx, dy)

        if self.old_estimate_pose is not None:
            #Calculate line between old and current. If the line has passed the target, increase the target
            self.check_jump(target_x,target_y)


        while self.jumped_past or distance < 1:
            #self.log_waypoint_data()
            self.jumped_past = False
            self.target_index += 1
            if self.target_index >= len(self.centerline):
                self.target_index = 0
                self.lap += 1
            target_x, target_y = self.centerline[self.target_index]
            dx = target_x - self.current_x
            dy = target_y - self.current_y
            distance = math.hypot(dx, dy)
            self.check_jump(target_x,target_y)


        desired_yaw = math.atan2(dy, dx)
        error_yaw = desired_yaw - self.current_yaw
        error_yaw = math.atan2(math.sin(error_yaw), math.cos(error_yaw))

        dt = (current_time - self.last_control_time).nanoseconds / 1e9
        self.last_control_time = current_time
        error_derivative = 0.0
        if dt > 1e-6:
            error_derivative = (error_yaw - self.previous_error_yaw) / dt
        
        self.previous_error_yaw = error_yaw


        kp = 0.50  # Adjust gain as needed
        kd= 0.06
        steering_angle = kp * error_yaw + kd * error_derivative

        msg = AckermannDriveStamped()

        self.old_estimate_pose = [self.current_x,self.current_y]


        if self.speed<=self.max_speed:
            self.speed+=0.05

        if self.lap >= 4:
            self.speed = 0

           #Stop the car after 4 laps and publish the final data
            self.log_waypoint_data(msg)
            self.get_logger().info("Stopping car after 4 laps")


        msg.drive.speed=self.speed/(1+abs(error_yaw/2.5))
        
        # if self.lap == 0:
        #     self.speed=1.3
        #     msg.drive.speed=1. Only necessary for real car testing

        msg.drive.steering_angle = steering_angle
        # print("x: ", self.current_x)
        # print("y: ", self.current_y)
        # print("yaw: ", self.current_yaw)
        if self.initizalized:
            self.publisher_.publish(msg)

    def check_jump(self,target_x,target_y):
            x_linspace = np.linspace(self.old_estimate_pose[0],self.current_x,5)
            y_linspace = np.linspace(self.old_estimate_pose[1],self.current_y,5)

                #Check if any distance was under 1:

            for pair in zip(x_linspace,y_linspace):
                    dx_line = math.hypot(target_x - pair[0], target_y-pair[1])
                    if dx_line < 1:
                        self.jumped_past = True
                        self._logger.info(f"Jumped past target: {pair[0]}, {pair[1]}")
                        break
            return

    def read_pgm(self,filename, byteorder='>'):
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

            #max size of width and height
            max_size = max(width, height)
            self.map_size = max_size

            # Convert to numpy array
            image = np.frombuffer(buffer, dtype='u1' if maxval < 256 else byteorder+'u2',
                                    count=int(width)*int(height), offset=len(header)).reshape((int(height), int(width)))

            coordinates_with_data = np.array(image, dtype=np.int32)
            
            image_ouput = np.full((max_size, max_size), 0)
            image_ouput = image_ouput.copy().astype(np.int32)
            # If image values are greater than occupied_thresh, set to 1, else 0
            image = image.copy().astype(float)

            #If image is not 1600x1600, pad it
            for i in range(len(image)):
                for j in range(len(image[i])):
                    if image[i, j] >=0.45:
                        image_ouput[i+max_size-1-height, j] = 0
                    else:
                        image_ouput[i+max_size-1-height, j] = 1
                        coordinates_with_data[i, j] = 1

            #if image is larger than 1600x1600, crop it

            #plot it


            return image, image_ouput

        except AttributeError:
            raise ValueError("Not a raw PGM file: '%s'" % filename)
        
    def get_map_info(self,map_path):
            #load origin, resolution and size from the map yaml
            with open(map_path, 'r') as f:
                map_yaml = yaml.safe_load(f)
                self.origin = map_yaml['origin']
                self.map_resolution = map_yaml['resolution']



class AckermannLineConvParent(AckermannLineParent):
    def __init__(self):
        AckermannLineParent.__init__(self)
        self.scan_sub = self.create_subscription(LaserScan, 'scan', self.scan_callback, 1)

        #self.lib = ctypes.CDLL('/home/f1t/test_cuda/cuda_test')
        
        #self.lib = ctypes.CDLL('/home/f1t/au_f1tenth_ws/control_loop/plotting/cuda.so')

        self.lib = ctypes.CDLL('/home/amandoee/control_loop/speeduprefined_dynamicmap.so')
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
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
        ]
        self.lib.convolve_lidar_scan_c_coarse_fine.restype = None
        self.map, self.coordinates_with_data = self.read_pgm(self.map_path+'.pgm')
        #plt.imshow(self.coordinates_with_data, cmap='gray',alpha=0.5)
        flipped_y_origin = (self.map_size/2 - abs(self.origin[1]/self.map_resolution))*2-self.origin[1]/self.map_resolution
        
        #plt.scatter(-self.origin[0] / self.map_resolution+self.centerline[0][0]/self.map_resolution, flipped_y_origin-self.centerline[0][1]/self.map_resolution, c='red', s=100)
        #plt.show()

        #print(self.xRange, self.yRange)



        startsize=100
        self.xRange = [max(round(flipped_y_origin-self.centerline[0][1]/self.map_resolution - startsize),0), min(round(flipped_y_origin-self.centerline[0][1]/self.map_resolution + startsize), self.map_size-1)]
        self.yRange = [max(round(-self.origin[0] / self.map_resolution+self.centerline[0][0]/self.map_resolution - startsize),0), min(round(-self.origin[0] / self.map_resolution+self.centerline[0][0]/self.map_resolution + startsize), self.map_size-1)]


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
                    ctypes.byref(best_angle),
                    int(self.map_size)
                )

                #If best angle is more different than 90 degrees, it is probably wrong, so we ignore it
                #print("Best angle: ", (float(best_angle.value) - 90))
                #print("Current yaw: ", np.rad2deg(self.current_yaw))

                #TODO: Check if rounding is correct.
                best_xy = np.where(result == best_sum.value)
                if len(best_xy[0]) > 1:
                    best_xy = (np.array([np.mean(best_xy[0])]), np.array([np.mean(best_xy[1])]))
                y_coord = (((self.map_size/2 - abs(self.origin[1] / self.map_resolution)) * 2 - self.origin[1] / self.map_resolution) - best_xy[0][0]) * self.map_resolution
                x_coord = (best_xy[1][0] + self.origin[0] / self.map_resolution) * self.map_resolution


                self.xRange = [max(0, int(best_xy[0][0] - self.rangesize)), min(self.map_size - 1, int(best_xy[0][0] + self.rangesize))]
                self.yRange = [max(0, int(best_xy[1][0] - self.rangesize)), min(self.map_size - 1, int(best_xy[1][0] + self.rangesize))]
                self._logger.info(f"X range: {self.xRange}")
                self._logger.info(f"Y range: {self.yRange}")

                # Update current pose based on the processed scan
                self.current_x = x_coord#(x_coord + self.current_x)/2
                self.current_y = y_coord#(y_coord + self.current_y)/2
                self.current_yaw = np.deg2rad(float(best_angle.value) - 90)
                self.initizalized = True
                # Publish a marker circle around the updated current pose
                self.publish_circle_marker(self.current_x, self.current_y)