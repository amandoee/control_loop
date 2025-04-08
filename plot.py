from ast import literal_eval
import re
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy import signal as sg
import cv2

resolution= 0.062500
origin= [-78.21853769831466,-44.37590462453829, 0.000000]
occupied_thresh= 0.45
free_thresh= 0.196

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


        #Convert to numpy array
        image = np.frombuffer(buffer, dtype='u1' if maxval < 256 else byteorder+'u2', count=int(width)*int(height), offset=len(header)).reshape((int(height), int(width)))

        coordinates_with_data = np.array(image,dtype=np.int32)
        
        

        #If image values are greater than occupied_thresh, set to 1, else 0
        image = image.copy().astype(float)
        for i in range(len(image)):
            for j in range(len(image[i])):
                if image[i,j] > 0.45:
                    image[i,j] = 0
                else:
                    image[i,j] = 1
                    coordinates_with_data[i,j] = 1


        return image, coordinates_with_data


    except AttributeError:
        raise ValueError("Not a raw PGM file: '%s'" % filename)
    
        
# Read CSV file
df = pd.read_csv('~/sim_ws/lidar_data.csv', sep=";")


def get_scan_from_index(index):

    # Initialize lists for trajectory and LiDAR points
    trajectory_x = []
    trajectory_y = []
    tracejectory_orientation = []

    all_lidar_points = []


    # Create plot


    x_coord = 0
    y_coord = 0
    x_lidar_local = []
    y_lidar_local = []
    orientation_from_index = 0

    row = df.iloc[index]
    # Convert to angle
    orientation = np.deg2rad(0)
    x = row[' Pose_X']
    y = row[' Pose_Y']
    trajectory_x.append(x)
    trajectory_y.append(y)
    tracejectory_orientation.append(orientation)
    
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
    valid = (valid_ranges >= range_min) & (valid_ranges <= 31)

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


def plot_result(output,coordinates_with_data):

    #If coordinates with data not None, plot the map:
    #if coordinates_with_data is not None:
    #    plt.imshow(coordinates_with_data, cmap='gray')
    #    plt.colorbar()


    #Only have the upper 99.9% of the values
    #max_val = np.percentile(output, 99.995)
    #output[output < max_val] = 0

    #Find the maximum value and circle it with blue
    max_val = np.max(output)
    max_index = np.where(output == max_val)
    plt.scatter(max_index[1], max_index[0], c='b', marker='o', label='Max value',alpha=0.5)

    #plt.imshow(output, cmap='jet',alpha=0.5)
    #plt.colorbar()
    #plot the coordinates with data


import ctypes
def convolve_lidar_scan(x_lidar_local, y_lidar_local,coordinates_with_data,orientation_in_rad,xRange,yRange):
    
    lib = ctypes.CDLL('./lidar_random_sample_log/speeduprefined.so')

    # Set argument types and return type for the C wrapper
    lib.convolve_lidar_scan_c_coarse_fine.argtypes = [
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
    ]
    lib.convolve_lidar_scan_c_coarse_fine.restype = None


    map_size=1600

    result = np.zeros((map_size,map_size), dtype=np.int32)


    best_sum = ctypes.c_int(0)
    sum = ctypes.c_int(-1)
    best_angle = ctypes.c_int(0)

    best_result = result.copy()

    best_orientation = orientation_in_rad


    x_lidar_local = np.round(x_lidar_local/resolution)
    y_lidar_local = np.round(y_lidar_local/resolution)


    # Remove duplicates by taking copy
    x_lidar_local_lowres = (x_lidar_local)
    y_lidar_local_lowres = (y_lidar_local)

    # Remove duplicates
    xy_lidar_local_lowres = np.array(list(set(zip(x_lidar_local_lowres,y_lidar_local_lowres))))
    print("Removed duplicates: ", len(x_lidar_local)-len(xy_lidar_local_lowres))

    x_lidar_local = np.ascontiguousarray(xy_lidar_local_lowres[:,0][::4])
    y_lidar_local = np.ascontiguousarray(xy_lidar_local_lowres[:,1][::4])


    #Measure the time it takes to calculate the best orientation
    import time
    start = time.time()

    

    lib.convolve_lidar_scan_c_coarse_fine(x_lidar_local, y_lidar_local, len(x_lidar_local), coordinates_with_data.ravel(), int(xRange[0]),int(xRange[1]),int(yRange[0]),int(yRange[1]),0,180, result.ravel(),ctypes.byref(sum),ctypes.byref(best_angle))

        
    if sum.value > best_sum.value:
        best_sum.value = sum.value
        best_result = result.copy()
        best_orientation = orientation_in_rad

    

    stop = time.time()
    print("Time to calculate best orientation: ", stop-start)
    
    plot_result(best_result,coordinates_with_data)
    
    #return (x,y, best_orientation)
    # find (x,y) of the best orientation
    best_xy = np.where(best_result == np.max(best_result))



    if len(best_xy[0]) > 1:
        print(best_xy)
        #Average the values
        best_xy = (np.array([np.mean(best_xy[0])]),np.array([np.mean(best_xy[1])]))
    print(best_xy)
        

    print("Best x,y: ", best_xy[1][0], best_xy[0][0])

    return best_xy[1][0], best_xy[0][0], best_orientation



#Random number for random lidar sample
import random
#Random number for random lidar sample

random_sample = random.randint(0, len(df)+100)
random_sample = 0

values = []
targets = []

for i in range(random_sample,random_sample+10):
    _, x_lidar_local, y_lidar_local,x_coord,y_coord, orientation_in_rad = get_scan_from_index(i)

    #plt.scatter(x_lidar_local[::8]/resolution, y_lidar_local[::8]/resolution, c='b', marker='o', label='LiDAR')

    flipped_y_origin = (800 - abs(origin[1]/resolution))*2-origin[1]/resolution

    #Plot the start position with orientation as arrow
    #plt.arrow(-origin[0]/resolution+x_coord/resolution, flipped_y_origin-y_coord/resolution, 0.5*np.cos(orientation_in_rad), 0.5*np.sin(orientation_in_rad), head_width=1, head_length=1, fc='r', ec='r')
    #plt.quiver(-origin[0]/resolution+x_coord/resolution, flipped_y_origin-y_coord/resolution, 0.5*np.cos(orientation_in_rad), 0.5*np.sin(orientation_in_rad), angles='xy', scale_units='xy', scale=0.001, color='r')
    
    plt.scatter(-origin[0]/resolution+x_coord/resolution, flipped_y_origin-y_coord/resolution, c='r', marker='x', label='Start position',alpha=0.5)
    print("Start position: ", -origin[0]/resolution+x_coord/resolution, flipped_y_origin-y_coord/resolution, orientation_in_rad)
    targets.append((-origin[0]/resolution+x_coord/resolution,flipped_y_origin-y_coord/resolution))
    #plt.legend()
    map, coordinates_with_data = read_pgm('./lidar_random_sample_log/maps/map0.pgm')


    xRangeStart = -origin[0]/resolution+x_coord/resolution-33
    xRangeEnd = -origin[0]/resolution+x_coord/resolution+33
    yRangeStart = flipped_y_origin-y_coord/resolution-33
    yRangeEnd = flipped_y_origin-y_coord/resolution+33

    x_val,y_val,orientation_val = convolve_lidar_scan(x_lidar_local=x_lidar_local, y_lidar_local=y_lidar_local,coordinates_with_data=coordinates_with_data,orientation_in_rad=orientation_in_rad,xRange=(yRangeStart,yRangeEnd),yRange=(xRangeStart,xRangeEnd))
    values.append((x_val,y_val,orientation_val))
#slider.on_changed(update)

#Calculate mean squared error
mse = 0
for i in range(len(values)):
    mse += (targets[i][0]-values[i][0])**2 + (targets[i][1]-values[i][1])**2

mse = mse/len(values)

print("Mean squared error: ", mse)

#Calculate mean absolute error
mae = 0
for i in range(len(values)):
    mae += abs(targets[i][0]-values[i][0]) + abs(targets[i][1]-values[i][1])

mae = mae/len(values)

print("Mean absolute error: ", mae)



#Plo




plt.imshow(coordinates_with_data, cmap='gray')
plt.show()


#Find the points with error larger than mse and plot them
for i in range(len(values)):
    if (targets[i][0]-values[i][0])**2 + (targets[i][1]-values[i][1])**2 > mse:
        plt.scatter(values[i][0],values[i][1],c='b',marker='o')
        plt.scatter(targets[i][0],targets[i][1],c='r',marker='x')
        
        #Plot the lidar scan
        _, x_lidar_local, y_lidar_local,x_coord,y_coord, orientation_in_rad = get_scan_from_index(i)


        #Rotate the lidar scan
        x_lidar_local_rotated = x_lidar_local*np.cos(orientation_in_rad) - y_lidar_local*np.sin(orientation_in_rad)
        y_lidar_local_rotated = x_lidar_local*np.sin(orientation_in_rad) + y_lidar_local*np.cos(orientation_in_rad) 


        y_lidar_local_rotated = -y_lidar_local_rotated


        plt.scatter(x_lidar_local_rotated/resolution+values[i][0], y_lidar_local_rotated/resolution+values[i][1], c='b', marker='o', label='LiDAR')
        

plt.imshow(coordinates_with_data, cmap='gray')
plt.show()

