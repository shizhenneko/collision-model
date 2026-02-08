#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
ROS Data Collection and Cleaning Script
---------------------------------------
Author: VOA Project Team
Description: 
    Collects and synchronizes data from Camera, Lidar, IMU, and Cmd_vel.
    Performs real-time cleaning and formatting:
    1. Image: Resize to 160x120.
    2. Lidar: Resample to fixed 360 points, clip ranges.
    3. IMU: Convert Quaternion to Euler, flatten to 9-dim vector.
    4. Cmd_vel: Asynchronous latest value.
    
    Output: data/{timestamp}/images/*.jpg and data.csv
"""

import rospy
import cv2
import csv
import os
import numpy as np
import time
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from tf.transformations import euler_from_quaternion

class DataCollector:
    def __init__(self):
        rospy.init_node('data_collector_node', anonymous=True)
        
        # --- Parameters ---
        self.output_root = rospy.get_param('~output_dir', 'data')
        self.target_size = (160, 120)  # (width, height)
        self.lidar_points = 360
        self.lidar_range_min = 0.05
        self.lidar_range_max = 8.0
        
        # --- State ---
        self.bridge = CvBridge()
        self.latest_cmd_vel = Twist()
        self.cmd_vel_timestamp = rospy.Time(0)
        self.recording = True
        self.frame_count = 0
        
        # --- Setup Output Directory ---
        timestamp_str = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(self.output_root, timestamp_str)
        self.image_dir = os.path.join(self.session_dir, 'images')
        
        if not os.path.exists(self.image_dir):
            os.makedirs(self.image_dir)
            
        self.csv_file_path = os.path.join(self.session_dir, 'data.csv')
        self.csv_file = open(self.csv_file_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Write CSV Header
        # imu_data format: [ax, ay, az, wx, wy, wz, roll, pitch, yaw]
        header = ['timestamp', 'image_path', 'lidar_ranges', 'imu_data', 'linear_x', 'angular_z']
        self.csv_writer.writerow(header)
        
        rospy.loginfo(f"Data collection started. Saving to: {self.session_dir}")
        
        # --- Subscribers ---
        # 1. Asynchronous Cmd_vel (Control Input)
        self.sub_cmd = rospy.Subscriber('/cmd_vel', Twist, self.cmd_vel_cb)
        
        # 2. Independent Sensors (Fallback Mode)
        self.latest_scan = None
        self.latest_imu = None
        
        self.sub_scan = rospy.Subscriber('/scan', LaserScan, self.scan_cb)
        self.sub_imu = rospy.Subscriber('/imu', Imu, self.imu_cb)
        
        # 3. Main Trigger (Camera)
        self.sub_image = rospy.Subscriber('/usb_cam/image_raw', Image, self.image_cb)
        
    def cmd_vel_cb(self, msg):
        """Cache the latest control command."""
        self.latest_cmd_vel = msg
        self.cmd_vel_timestamp = rospy.Time.now()

    def scan_cb(self, msg):
        """Cache latest Lidar scan."""
        self.latest_scan = msg

    def imu_cb(self, msg):
        """Cache latest IMU data."""
        self.latest_imu = msg

    def process_image(self, msg):
        """Convert ROS Image to resized OpenCV image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        except CvBridgeError as e:
            rospy.logerr(f"CV Bridge Error: {e}")
            return None

        # Resize to 160x120
        resized_image = cv2.resize(cv_image, self.target_size, interpolation=cv2.INTER_AREA)
        return resized_image

    def process_lidar(self, msg):
        """Resample Lidar scan to fixed 360 points and clean data."""
        ranges = np.array(msg.ranges)
        
        # Handle Inf/NaN: Replace with max_range (or a large number indicating no obstacle)
        # Using msg.range_max is safer, but sometimes it's inf. Use self.lidar_range_max.
        ranges = np.nan_to_num(ranges, posinf=self.lidar_range_max, neginf=0.0)
        
        # Clip to valid range [min, max]
        ranges = np.clip(ranges, self.lidar_range_min, self.lidar_range_max)
        
        # Resample to 360 points
        # Current angles
        current_angles = np.linspace(msg.angle_min, msg.angle_max, len(ranges))
        
        # Target angles (360 points over the same FOV)
        # Note: If the lidar is 360 degrees, angle_max - angle_min should be approx 2*pi
        target_angles = np.linspace(msg.angle_min, msg.angle_max, self.lidar_points)
        
        # Linear interpolation
        resampled_ranges = np.interp(target_angles, current_angles, ranges)
        
        # Return as list for JSON serialization
        return resampled_ranges.tolist()

    def process_imu(self, msg):
        """Extract Acc, Gyro and convert Quat to Euler."""
        # 1. Linear Acceleration
        ax = msg.linear_acceleration.x
        ay = msg.linear_acceleration.y
        az = msg.linear_acceleration.z
        
        # 2. Angular Velocity
        wx = msg.angular_velocity.x
        wy = msg.angular_velocity.y
        wz = msg.angular_velocity.z
        
        # 3. Orientation (Quaternion -> Euler)
        orientation_q = msg.orientation
        quat_list = [orientation_q.x, orientation_q.y, orientation_q.z, orientation_q.w]
        (roll, pitch, yaw) = euler_from_quaternion(quat_list)
        
        # Combine into 9-dim vector
        return [ax, ay, az, wx, wy, wz, roll, pitch, yaw]

    def image_cb(self, image_msg):
        """Main trigger callback (Fallback/Async Mode)."""
        if not self.recording:
            return

        # Fallback Check: Do we have other sensor data?
        if self.latest_scan is None or self.latest_imu is None:
            # Wait for initialization
            return

        # Use image timestamp as the primary key
        timestamp = image_msg.header.stamp
        timestamp_ns = timestamp.to_nsec()
        
        # 1. Process Image
        processed_img = self.process_image(image_msg)
        if processed_img is None:
            return
            
        # Save Image
        img_filename = f"{timestamp_ns}.jpg"
        img_path_rel = os.path.join('images', img_filename)
        img_path_abs = os.path.join(self.image_dir, img_filename)
        cv2.imwrite(img_path_abs, processed_img)
        
        # 2. Process Lidar (Use Cached)
        lidar_data = self.process_lidar(self.latest_scan)
        
        # 3. Process IMU (Use Cached)
        imu_data = self.process_imu(self.latest_imu)
        
        # 4. Get latest Cmd_vel
        # Optional: check if cmd_vel is stale (e.g. > 0.5s old)
        # For now, we trust the latest command is the active intent.
        v = self.latest_cmd_vel.linear.x
        w = self.latest_cmd_vel.angular.z
        
        # 5. Write to CSV
        # lidar_data and imu_data are lists, convert to string representation
        self.csv_writer.writerow([
            timestamp_ns,
            img_path_rel,
            str(lidar_data),
            str(imu_data),
            v,
            w
        ])
        
        # Log periodically
        self.frame_count += 1
        if self.frame_count % 100 == 0:
            rospy.loginfo(f"Recorded frame at {timestamp.to_sec():.2f}")

    def shutdown(self):
        rospy.loginfo("Stopping data collection...")
        self.csv_file.close()
        rospy.loginfo(f"Data saved to {self.session_dir}")

if __name__ == '__main__':
    try:
        collector = DataCollector()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        if 'collector' in locals():
            collector.shutdown()
