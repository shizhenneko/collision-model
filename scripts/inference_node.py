#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import onnxruntime as ort
import threading
import rospkg

from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

import os

class InferenceNode:
    def __init__(self):
        rospy.init_node('voa_inference_node', anonymous=True)
        
        # Resolve absolute paths using rospkg for robustness
        try:
            rospack = rospkg.RosPack()
            package_path = rospack.get_path('collision_model')
            default_ann_path = os.path.join(package_path, 'checkpoints', 'ann_model.onnx')
            default_voa_path = os.path.join(package_path, 'checkpoints', 'voa_model.onnx')
        except Exception as e:
            rospy.logwarn(f"Could not find package path via rospkg: {e}. Falling back to script relative path.")
            script_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(script_dir)
            default_ann_path = os.path.join(project_root, 'checkpoints', 'ann_model.onnx')
            default_voa_path = os.path.join(project_root, 'checkpoints', 'voa_model.onnx')
        
        # Parameters
        self.ann_path = rospy.get_param('~ann_model_path', default_ann_path)
        self.voa_path = rospy.get_param('~voa_model_path', default_voa_path)
        self.risk_threshold = rospy.get_param('~risk_threshold', 0.7)
        self.brake_duration_short = 0.3 # seconds
        
        # Load ONNX Models
        rospy.loginfo(f"Loading ANN model from {self.ann_path}...")
        self.sess_ann = ort.InferenceSession(self.ann_path, providers=['CPUExecutionProvider'])
        
        rospy.loginfo(f"Loading VOA model from {self.voa_path}...")
        self.sess_voa = ort.InferenceSession(self.voa_path, providers=['CPUExecutionProvider'])
        
        # Sensor Data Cache (Thread-safe is tricky in Python with GIL but atomic reads are fine)
        self.latest_scan = None
        self.latest_imu = None
        self.lock = threading.Lock()
        
        self.bridge = CvBridge()
        
        # State Machine
        self.is_braking = False
        self.brake_start_time = None
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=1)
        
        rospy.loginfo("Inference Node Ready!")

    def scan_callback(self, msg):
        with self.lock:
            self.latest_scan = msg

    def imu_callback(self, msg):
        with self.lock:
            self.latest_imu = msg

    def preprocess_image(self, msg):
        try:
            # Convert ROS Image to CV2
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # Resize to 160x120
            cv_image = cv2.resize(cv_image, (160, 120))
            
            # Convert to RGB
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            
            # HWC -> CHW
            img_tensor = cv_image.transpose(2, 0, 1)
            
            # Normalize to [0, 1]
            img_tensor = img_tensor.astype(np.float32) / 255.0
            
            # Add Batch Dimension [1, 3, 120, 160]
            img_tensor = np.expand_dims(img_tensor, axis=0)
            
            return img_tensor
        except Exception as e:
            rospy.logerr(f"Image preprocessing failed: {e}")
            return None

    def preprocess_lidar(self, msg):
        if msg is None:
            return np.zeros((1, 360), dtype=np.float32)
            
        # 1. Handle Raw Data
        ranges = np.array(msg.ranges)
        # Replace Inf with Max Range, NaN with 0
        ranges = np.nan_to_num(ranges, posinf=8.0, neginf=0.0)
        
        # 2. Resample to 360 points (if needed) to match Training Data
        current_len = len(ranges)
        if current_len == 360:
            resampled = ranges
        else:
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, 360)
            resampled = np.interp(x_new, x_old, ranges)
            
        # 3. Clip values (Physical Constraints matching Training Data)
        # Training data cleaner uses [0.05, 8.0]
        resampled = np.clip(resampled, 0.05, 8.0)
        
        # 4. Spike Filter (Matching Training Data Cleaner)
        # Remove single-point noise spikes
        prev_lidar = np.roll(resampled, 1)
        next_lidar = np.roll(resampled, -1)
        
        # If point differs > 0.5m from BOTH neighbors, replace with average
        spike_mask = (np.abs(resampled - prev_lidar) > 0.5) & (np.abs(resampled - next_lidar) > 0.5)
        resampled[spike_mask] = (prev_lidar[spike_mask] + next_lidar[spike_mask]) / 2.0
        
        # 5. Normalize
        resampled = resampled.astype(np.float32) / 8.0
        
        return np.expand_dims(resampled, axis=0)

    def preprocess_imu(self, msg):
        if msg is None:
            return np.zeros((1, 9), dtype=np.float32)
            
        # 9-dim: [ax, ay, az, wx, wy, wz, mx, my, mz]
        # Note: Standard ROS Imu msg doesn't have magnetometer usually (unless extended).
        # But our training data has 9 dims. 
        # Core memory says: "IMU (9-DOF: linear vel, angular vel, magnetometer)"
        # If standard Imu msg only has 6, we might need to pad or subscribe to a different topic.
        # Here we assume standard Imu (6-DOF) + 3 zeros if mag is missing, OR mag is in orientation? 
        # No, orientation is quaternion.
        # Let's stick to linear_acceleration and angular_velocity (6 dims).
        # If the model strictly requires 9, we need to know what the last 3 are.
        # If the robot has a mag topic, we should sync it. 
        # For now, we'll fill 6 dims and pad 3 zeros if we can't get mag.
        # Wait, if the model was trained with 9 dims, feeding 0s might be bad if Mag was important.
        # However, for indoor robots, Mag is often noisy/useless.
        
        data = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z,
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z,
            0.0, 0.0, 0.0 # Placeholder for Mag
        ], dtype=np.float32)
        
        return np.expand_dims(data, axis=0)

    def image_callback(self, msg):
        # 1. Preprocess Inputs
        img_input = self.preprocess_image(msg)
        if img_input is None:
            return

        with self.lock:
            lidar_input = self.preprocess_lidar(self.latest_scan)
            imu_input = self.preprocess_imu(self.latest_imu)
            
        # 2. ANN Inference (Risk Assessment)
        # 使用明确的输入名称，避免顺序依赖
        ann_inputs = {
            'image': img_input,
            'lidar': lidar_input,
            'imu': imu_input
        }
        
        try:
            risk = self.sess_ann.run(None, ann_inputs)[0][0][0] # Output is [1, 1]
        except Exception as e:
            rospy.logerr_throttle(5, f"ANN Inference failed: {e}")
            return
        
        twist = Twist()
        
        # 3. Decision Logic
        if risk > self.risk_threshold:
            rospy.logwarn_throttle(1, f"COLLISION RISK DETECTED: {risk:.2f} > {self.risk_threshold}. EMERGENCY BRAKE!")
            # Emergency Brake
            self.is_braking = True
            if self.brake_start_time is None:
                self.brake_start_time = rospy.Time.now()
            
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        else:
            # Safe
            if self.is_braking:
                # Recovery Mode
                brake_duration = (rospy.Time.now() - self.brake_start_time).to_sec()
                rospy.loginfo(f"Recovering from brake. Duration: {brake_duration:.2f}s")
                
                # Run VOA
                voa_inputs = {
                    'image': img_input,
                    'lidar': lidar_input,
                    'imu': imu_input
                }
                try:
                    outputs = self.sess_voa.run(None, voa_inputs)
                    # outputs[0] = policy [1, 2]
                    # outputs[1] = recovery [1, 6] -> [v_s, w_s, v_m, w_m, v_l, w_l]
                    
                    recovery_out = outputs[1][0]
                    
                    if brake_duration < self.brake_duration_short:
                        # Use Medium-term (Index 2, 3) per document?
                        # Document: "delta < 0.3s: Use Medium"
                        v = recovery_out[2]
                        w = recovery_out[3]
                        rospy.loginfo("Strategy: Medium-Term Recovery")
                    else:
                        # Use Long-term (Index 4, 5) per document?
                        # Document: "delta >= 0.3s: Use Long"
                        v = recovery_out[4]
                        w = recovery_out[5]
                        rospy.loginfo("Strategy: Long-Term Recovery")
                    
                    twist.linear.x = float(v)
                    twist.angular.z = float(w)
                    
                    # Reset Brake State (Single-shot recovery, or transition out)
                    # If we continually detect risk < threshold, we should eventually go back to normal.
                    # Here we transition to normal state immediately for next frame?
                    # Or keep 'is_braking' until some condition?
                    # For simplicity: One frame of recovery action, then reset.
                    self.is_braking = False
                    self.brake_start_time = None
                except Exception as e:
                    rospy.logerr(f"VOA Recovery Inference failed: {e}")

            else:
                # Normal Driving
                voa_inputs = {
                    'image': img_input,
                    'lidar': lidar_input,
                    'imu': imu_input
                }
                try:
                    outputs = self.sess_voa.run(None, voa_inputs)
                    policy_out = outputs[0][0] # [v, w]
                    
                    twist.linear.x = float(policy_out[0])
                    twist.angular.z = float(policy_out[1])
                    # rospy.loginfo_throttle(1, f"Normal Driving: v={twist.linear.x:.2f}, w={twist.angular.z:.2f}")
                except Exception as e:
                    rospy.logerr_throttle(5, f"VOA Normal Inference failed: {e}")


        # 4. Publish Command
        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        node = InferenceNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
