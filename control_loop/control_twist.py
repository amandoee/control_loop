#!/usr/bin/env python

# Author: christoph.roesmann@tu-dortmund.de
# Modifier: Donghee Han, hdh7485@kaist.ac.kr

import rclpy, math
from geometry_msgs.msg import Twist
from ackermann_msgs.msg import AckermannDriveStamped
from ackermann_msgs.msg import AckermannDrive


def convert_trans_rot_vel_to_steering_angle(v, omega, wheelbase):
  if omega == 0 or v == 0:
    return 0

  radius = v / omega
  return math.atan(wheelbase / radius)


def cmd_callback(data):
  global wheelbase
  global ackermann_cmd_topic
  global frame_id
  global pub
  global message_type
  
  if message_type == 'ackermann_drive':
    v = data.linear.x
    steering = convert_trans_rot_vel_to_steering_angle(v, data.angular.z, wheelbase)
    
    msg = AckermannDrive()
    msg.steering_angle = steering
    msg.speed = v
    
    pub.publish(msg)

  else:
    v = data.linear.x
    steering = convert_trans_rot_vel_to_steering_angle(v, data.angular.z, wheelbase)
    
    msg = AckermannDriveStamped()
    msg.header.stamp = rclpy.Time.now()
    msg.header.frame_id = frame_id
    msg.drive.steering_angle = steering
    msg.drive.speed = v
    
    pub.publish(msg)



def main():
  try:
    
    rclpy.init_node('cmd_vel_to_ackermann_drive')
        
    twist_cmd_topic = rclpy.get_param('~twist_cmd_topic', '/cmd_vel') 
    ackermann_cmd_topic = rclpy.get_param('~ackermann_cmd_topic', '/ackermann_cmd')
    wheelbase = rclpy.get_param('~wheelbase', 1.0)
    frame_id = rclpy.get_param('~frame_id', 'odom')
    message_type = rclpy.get_param('~message_type', 'ackermann_drive') # ackermann_drive or ackermann_drive_stamped
    
    rclpy.Subscriber(twist_cmd_topic, Twist, cmd_callback, queue_size=1)
    if message_type == 'ackermann_drive':
      pub = rclpy.Publisher(ackermann_cmd_topic, AckermannDrive, queue_size=1)
    else:
      pub = rclpy.Publisher(ackermann_cmd_topic, AckermannDriveStamped, queue_size=1)
    
    rclpy.loginfo("Node 'cmd_vel_to_ackermann_drive' started.\nListening to %s, publishing to %s. Frame id: %s, wheelbase: %f", "/cmd_vel", ackermann_cmd_topic, frame_id, wheelbase)
    
    rclpy.spin()
    
  except rclpy.ROSInterruptException:
    pass

if __name__ == '__main__': 
  main()