from ast import literal_eval
import ctypes
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sg
import cv2
import yaml
import random


class InitialposeEstimator:
    def __init__(self, mapstring,datastring):
        print("init")
        self.df = pd.read_csv(datastring, sep=";")
        #random number between 0 and 1
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
        self.map, self.coordinates_with_data = self.read_pgm(mapstring+'.pgm')
    
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
        


# Read CSV file

    def get_scan_from_index(self,index):
        all_lidar_points = []


        # Create plot

        x_lidar_local = []
        y_lidar_local = []
        orientation_from_index = 0

        row = self.df.iloc[index]
        # Convert to angle
        x = row[' Pose_X']
        y = row[' Pose_Y']

        # Parse LaserScan data
        scan_str = row[' Lidar_Ranges_Count']
        
        try:
            angle_min = float(re.search(r'angle_min=([-\d.]+)', scan_str).group(1))
            angle_max = float(re.search(r'angle_max=([-\d.]+)', scan_str).group(1))
            angle_inc = float(re.search(r'angle_increment=([-\d.]+)', scan_str).group(1))
            range_min = float(re.search(r'range_min=([\d.]+)', scan_str).group(1))
            range_max = float(re.search(r'range_max=([\d.]+)', scan_str).group(1))
            
            ranges_str = re.search(r'ranges=\[([^\]]+)\]', scan_str).group(1)
            ranges = literal_eval(f'[{ranges_str}]')
        except (AttributeError, ValueError) as e:
            print(f"Error parsing row: {e}")
            
        angles = np.linspace(angle_min, angle_max, len(ranges))
        valid_ranges = np.array(ranges)
        valid = (valid_ranges >= range_min) & (valid_ranges <= 29.9)

        y_lidar = valid_ranges[valid] * np.sin(angles[valid])
        x_lidar = valid_ranges[valid] * np.cos(angles[valid])
        
    
        x_lidar_local = x_lidar
        y_lidar_local = y_lidar
        x_coord = x
        y_coord = y
        orientation_from_index = np.deg2rad(row[' Orientation'])
        
        x_global = x_lidar + x
        y_global = y_lidar + y
        
        
        all_lidar_points.append((x_global, y_global))

        return all_lidar_points, x_lidar_local, y_lidar_local, x_coord, y_coord, orientation_from_index

    def get_map_info(self,map_path):
            #load origin, resolution and size from the map yaml
            with open(map_path, 'r') as f:
                map_yaml = yaml.safe_load(f)
                self.origin = map_yaml['origin']
                self.map_resolution = map_yaml['resolution']



    def plot_result(self,coordinates_with_data):
        #Random number for random lidar sample
        #Random number for random lidar sample

        #random_sample = random.randint(0, len(self.df)-1)
        random_sample = 1
        values = []
        targets = []

        for i in range(random_sample,random_sample+1):
            _, x_lidar_local, y_lidar_local,x_coord,y_coord, orientation_in_rad = self.get_scan_from_index(i)

            flipped_y_origin = (self.map_size/2 - abs(self.origin[1]/self.map_resolution))*2-self.origin[1]/self.map_resolution


            plt.scatter(-self.origin[0]/self.map_resolution+x_coord/self.map_resolution, flipped_y_origin-y_coord/self.map_resolution, c='r', marker='x', label='Start position',alpha=0.5)
            print("Start position: ", -self.origin[0]/self.map_resolution+x_coord/self.map_resolution, flipped_y_origin-y_coord/self.map_resolution, orientation_in_rad)
            targets.append((-self.origin[0]/self.map_resolution+x_coord/self.map_resolution,flipped_y_origin-y_coord/self.map_resolution))

            #PLOT THE LIDAR SCAN ROTATED
            x_lidar_local_rotated = x_lidar_local*np.cos(orientation_in_rad) - y_lidar_local*np.sin(orientation_in_rad)
            y_lidar_local_rotated = x_lidar_local*np.sin(orientation_in_rad) + y_lidar_local*np.cos(orientation_in_rad)
            plt.scatter(-self.origin[0]/self.map_resolution+x_lidar_local_rotated/self.map_resolution+x_coord/self.map_resolution, flipped_y_origin-(y_lidar_local_rotated/self.map_resolution+y_coord/self.map_resolution), c='b', marker='o', label='LiDAR')

            x_val,y_val,orientation_val = self.scan_callback(x_lidar_local=x_lidar_local, y_lidar_local=y_lidar_local)

            #Account for the origin
            print("Estimated position: ", x_val, y_val, orientation_val)
            plt.scatter(x_val, y_val, c='g', marker='o', label='Estimated position')

            #Plot the estimated position with the lidar scan
            x_lidar_local_rotated = x_lidar_local*np.cos(orientation_val) - y_lidar_local*np.sin(orientation_val)
            y_lidar_local_rotated = x_lidar_local*np.sin(orientation_val) + y_lidar_local*np.cos(orientation_val)
            plt.scatter(x_val+x_lidar_local_rotated/self.map_resolution, y_val+(-y_lidar_local_rotated)/self.map_resolution, c='g', marker='o', label='Estimated LiDAR scan')
            plt.legend()

            #values.append((x_val,y_val,orientation_val))
        #slider.on_changed(update)

        #Calculate mean squared error
        mse = 0
        for i in range(len(values)):
            mse += (targets[i][0]-values[i][0])**2 + (targets[i][1]-values[i][1])**2

        #mse = mse/len(values)

        print("Mean squared error: ", mse)

        #Calculate mean absolute error
        mae = 0
        for i in range(len(values)):
            mae += abs(targets[i][0]-values[i][0]) + abs(targets[i][1]-values[i][1])

        #mae = mae/len(values)

        print("Mean absolute error: ", mae)

        plt.imshow(coordinates_with_data, cmap='gray',alpha=0.5)
        plt.show()


        #Find the points with error larger than mse and plot them
        for i in range(len(values)):
            if (targets[i][0]-values[i][0])**2 + (targets[i][1]-values[i][1])**2 > mse:
                plt.scatter(values[i][0],values[i][1],c='b',marker='o')
                plt.scatter(targets[i][0],targets[i][1],c='r',marker='x')
                
                #Plot the lidar scan
                _, x_lidar_local, y_lidar_local,x_coord,y_coord, orientation_in_rad = self.get_scan_from_index(i)


                #Rotate the lidar scan
                x_lidar_local_rotated = x_lidar_local*np.cos(orientation_in_rad) - y_lidar_local*np.sin(orientation_in_rad)
                y_lidar_local_rotated = x_lidar_local*np.sin(orientation_in_rad) + y_lidar_local*np.cos(orientation_in_rad) 


                y_lidar_local_rotated = -y_lidar_local_rotated


                plt.scatter(x_lidar_local_rotated/self.map_resolution+values[i][0], y_lidar_local_rotated/self.map_resolution+values[i][1], c='b', marker='o', label='LiDAR')
                



    def scan_callback(self, x_lidar_local, y_lidar_local):
                # Store the latest lidar scan

                result = np.zeros((self.map_size, self.map_size), dtype=np.int32)
                best_sum = ctypes.c_int(0)
                best_angle = ctypes.c_int(0)

                xy_lidar_local_lowrest = np.array(list(set(zip(x_lidar_local, y_lidar_local))))
                x_lidar_local = np.ascontiguousarray(xy_lidar_local_lowrest[:, 0][::4])
                y_lidar_local = np.ascontiguousarray(xy_lidar_local_lowrest[:, 1][::4])

                

                self.lib.convolve_lidar_scan_c_coarse_fine(
                        x_lidar_local,
                        y_lidar_local,
                        len(x_lidar_local),
                        self.coordinates_with_data.ravel(),
                        0,
                        self.map_size-1,
                        0,
                        self.map_size-1,
                        0,
                        180,
                        result.ravel(),
                        ctypes.byref(best_sum),
                        ctypes.byref(best_angle),
                        int(self.map_size)
                )

                    #If best angle is more different than 90 degrees, it is probably wrong, so we ignore it
                    #print("Best angle: ", (float(best_angle.value) - 90))
                    #print("Current yaw: ", np.rad2deg(self.current_yaw))

                #TODO: Check if rounding is correct.
                best_xy = np.where(result == np.max(result))
                if len(best_xy[0]) > 1:
                    best_xy = (np.array([np.mean(best_xy[0])]), np.array([np.mean(best_xy[1])]))

                print("result says: ", best_xy[1][0], best_xy[0][0], np.deg2rad(float(best_angle.value) - 90))
                #self.xRange = [max(0, int(best_xy[0][0] - self.rangesize)), min(self.map_size - 1, int(best_xy[0][0] + self.rangesize))]
                #self.yRange = [max(0, int(best_xy[1][0] - self.rangesize)), min(self.map_size - 1, int(best_xy[1][0] + self.rangesize))]
                #self._logger.info(f"X range: {self.xRange}")
                #self._logger.info(f"Y range: {self.yRange}")
                # Update current pose based on the processed scan

                return best_xy[1][0], best_xy[0][0], np.deg2rad(float(best_angle.value) - 90)


def main ():
    map_path = '/home/amandoee/control_loop/maps/map0'
    data_path = '/home/amandoee/control_loop/plotting/lidar_data_recent.csv'

    initialpose_estimator = InitialposeEstimator(map_path, data_path)
    initialpose_estimator.get_map_info(map_path+'.yaml')
    initialpose_estimator.plot_result(initialpose_estimator.coordinates_with_data)

if __name__ == "__main__":
    main()