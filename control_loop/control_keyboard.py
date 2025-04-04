#!/usr/bin/env python
import rclpy

from ackermann_msgs.msg import AckermannDriveStamped  # Changed to AckermannDriveStamped
from std_msgs.msg import Header  # Header for AckermannDriveStamped

import sys, select, termios, tty

settings = termios.tcgetattr(sys.stdin)

msg = """
Reading from the keyboard and Publishing to AckermannDriveStamped!
---------------------------
Moving around:
   u : left forward
   i : forward
   o : right forward
   m : left backward
   , : backward
   . : right backward

q/z : increase/decrease max speed by 10%
w/x : increase/decrease only linear speed by 10%
e/c : increase/decrease only steering angle by 10%

CTRL-C to quit
"""

moveBindings = {
    'u': (1, 1),   # left forward
    'i': (1, 0),   # forward
    'o': (1, -1),  # right forward
    'm': (-1, 1),  # left backward
    ',': (-1, 0),  # backward
    '.': (-1, -1), # right backward
}

speedBindings = {
    'q': (1.1, 1.1),
    'z': (0.9, 0.9),
    'w': (1.1, 1),
    'x': (0.9, 1),
    'e': (1, 1.1),
    'c': (1, 0.9),
}

def getKey():
    tty.setraw(sys.stdin.fileno())
    key = sys.stdin.read(1)  # Directly read the key input without waiting for select
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)
    return key


def vels(speed, steering):
    return f"currently:\tspeed {speed}\tsteering {steering}"

def main(args=None):
    if args is None:
        args = sys.argv

    rclpy.init()
    node = rclpy.create_node('teleop_ackermann_keyboard')
    
    pub = node.create_publisher(AckermannDriveStamped, '/drive', 1)
    
    speed = 3.0
    steering = 0.5
    x = 0
    th = 0
    status = 0

    try:
        print(msg)
        print(vels(speed, steering))
        while True:
            key = getKey()
            if key in moveBindings.keys():
                x = moveBindings[key][0]
                th = moveBindings[key][1]
            elif key in speedBindings.keys():
                speed *= speedBindings[key][0]
                steering *= speedBindings[key][1]
                print(vels(speed, steering))
                if status == 14:
                    print(msg)
                status = (status + 1) % 15
            else:
                x = 0
                th = 0
                if key == '\x03':
                    break

            # Create AckermannDriveStamped message
            drive_msg = AckermannDriveStamped()
            drive_msg.header = Header()  # Create header
            drive_msg.header.stamp = node.get_clock().now().to_msg()  # Timestamp the message
            drive_msg.header.frame_id = 'base_link'  # Add frame ID if necessary (can adjust as per requirement)
            drive_msg.drive.speed = x * speed
            drive_msg.drive.steering_angle = th * steering
            pub.publish(drive_msg)
    
    except Exception as e:
        print(e)
    
    finally:
        drive_msg = AckermannDriveStamped()
        drive_msg.header = Header()
        drive_msg.header.stamp = node.get_clock().now().to_msg()
        drive_msg.header.frame_id = 'base_link'
        drive_msg.drive.speed = 0.0
        drive_msg.drive.steering_angle = 0.0
        pub.publish(drive_msg)
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, settings)

if __name__ == '__main__':  # This part ensures main() is called only when running directly
    main()
