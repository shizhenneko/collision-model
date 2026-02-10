#!/usr/bin/env python3
import rospy
import cv2
import numpy as np
import onnxruntime as ort
import threading
import math
import tf.transformations
import rospkg

from sensor_msgs.msg import Image, LaserScan, Imu
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from cv_bridge import CvBridge

import os

class NavigationNode:
    def __init__(self):
        rospy.init_node('navigation_node', anonymous=True)
        
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
        self.risk_threshold = rospy.get_param('~risk_threshold', 0.6) # Lowered slightly for earlier blending
        self.brake_duration_short = 0.3
        self.goal_tolerance = 0.2
        
        # Load ONNX Models
        rospy.loginfo(f"Loading ANN model from {self.ann_path}...")
        self.sess_ann = ort.InferenceSession(self.ann_path, providers=['CPUExecutionProvider'])
        
        rospy.loginfo(f"Loading VOA model from {self.voa_path}...")
        self.sess_voa = ort.InferenceSession(self.voa_path, providers=['CPUExecutionProvider'])
        
        # Sensor Data Cache
        self.latest_scan = None
        self.latest_imu = None
        self.current_pose = None
        self.current_goal = None
        self.lock = threading.Lock()
        
        self.bridge = CvBridge()
        
        # State Machine
        self.is_braking = False
        self.brake_start_time = None
        self.nav_state = "IDLE" # IDLE, NAVIGATING, REACHED

        # Avoidance Persistence (Anti-Oscillation)
        self.last_avoidance_time = rospy.Time.now()
        self.avoidance_cooldown = 2.0 # Seconds to suppress Nav after VOA action
        
        # Stuck Detection
        self.stuck_check_interval = 2.0
        self.last_stuck_check_time = rospy.Time.now()
        self.last_pose_position = None
        self.stuck_distance_threshold = 0.1 # m
        self.is_recovering = False
        self.recovery_start_time = None
        
        # Publishers
        self.cmd_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        
        # Subscribers
        rospy.Subscriber('/camera/image_raw', Image, self.image_callback, queue_size=1, buff_size=2**24)
        rospy.Subscriber('/scan', LaserScan, self.scan_callback, queue_size=1)
        rospy.Subscriber('/imu/data', Imu, self.imu_callback, queue_size=1)
        rospy.Subscriber('/odom', Odometry, self.odom_callback, queue_size=1)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback, queue_size=1)
        
        rospy.loginfo("Navigation Node Ready! Waiting for Goal...")

    def scan_callback(self, msg):
        with self.lock:
            self.latest_scan = msg

    def imu_callback(self, msg):
        with self.lock:
            self.latest_imu = msg
            
    def odom_callback(self, msg):
        with self.lock:
            self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        with self.lock:
            self.current_goal = msg.pose
            self.nav_state = "NAVIGATING"
            self.is_recovering = False # Reset recovery
            rospy.loginfo("New Goal Received!")

    def check_if_stuck(self, cmd_v, cmd_w):
        # Only check if we are trying to move
        if abs(cmd_v) < 0.05 and abs(cmd_w) < 0.1:
            return False
            
        now = rospy.Time.now()
        if (now - self.last_stuck_check_time).to_sec() > self.stuck_check_interval:
            is_stuck = False
            if self.current_pose is not None and self.last_pose_position is not None:
                # Calc distance moved
                dx = self.current_pose.position.x - self.last_pose_position.x
                dy = self.current_pose.position.y - self.last_pose_position.y
                dist_moved = math.sqrt(dx*dx + dy*dy)
                
                if dist_moved < self.stuck_distance_threshold:
                    is_stuck = True
            
            if self.current_pose is not None:
                self.last_pose_position = self.current_pose.position
                
            self.last_stuck_check_time = now
            return is_stuck
        return False

    def preprocess_image(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            cv_image = cv2.resize(cv_image, (160, 120))
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_tensor = cv_image.transpose(2, 0, 1)
            img_tensor = img_tensor.astype(np.float32) / 255.0
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
        ranges = np.nan_to_num(ranges, posinf=8.0, neginf=0.0)
        
        # 2. Resample to 360 points
        current_len = len(ranges)
        if current_len == 360:
            resampled = ranges
        else:
            x_old = np.linspace(0, 1, current_len)
            x_new = np.linspace(0, 1, 360)
            resampled = np.interp(x_new, x_old, ranges)
            
        # 3. Clip (Match Training)
        resampled = np.clip(resampled, 0.05, 8.0)
        
        # 4. Spike Filter (Match Training)
        prev_lidar = np.roll(resampled, 1)
        next_lidar = np.roll(resampled, -1)
        spike_mask = (np.abs(resampled - prev_lidar) > 0.5) & (np.abs(resampled - next_lidar) > 0.5)
        resampled[spike_mask] = (prev_lidar[spike_mask] + next_lidar[spike_mask]) / 2.0
        
        # 5. Normalize
        resampled = resampled.astype(np.float32) / 8.0
        return np.expand_dims(resampled, axis=0)

    def preprocess_imu(self, msg):
        if msg is None:
            return np.zeros((1, 9), dtype=np.float32)
        data = np.array([
            msg.linear_acceleration.x, msg.linear_acceleration.y, msg.linear_acceleration.z,
            msg.angular_velocity.x, msg.angular_velocity.y, msg.angular_velocity.z,
            0.0, 0.0, 0.0
        ], dtype=np.float32)
        return np.expand_dims(data, axis=0)

    def get_nav_cmd(self):
        if self.current_goal is None or self.current_pose is None:
            return 0.0, 0.0
            
        # Extract positions
        gx, gy = self.current_goal.position.x, self.current_goal.position.y
        px, py = self.current_pose.position.x, self.current_pose.position.y
        
        # Extract Yaw
        q = self.current_pose.orientation
        _, _, yaw = tf.transformations.euler_from_quaternion([q.x, q.y, q.z, q.w])
        
        # Distance and Heading
        dx, dy = gx - px, gy - py
        dist = math.sqrt(dx*dx + dy*dy)
        target_heading = math.atan2(dy, dx)
        
        # Heading Error
        heading_error = target_heading - yaw
        # Normalize to [-pi, pi]
        while heading_error > math.pi: heading_error -= 2*math.pi
        while heading_error < -math.pi: heading_error += 2*math.pi
        
        # Check Goal Reached
        if dist < self.goal_tolerance:
            self.nav_state = "REACHED"
            rospy.loginfo("Goal Reached!")
            return 0.0, 0.0
            
        # P Controller
        v_cmd = min(dist * 0.5, 0.3) # Max speed 0.3 m/s
        w_cmd = 1.5 * heading_error
        
        # Limit w
        w_cmd = max(min(w_cmd, 1.0), -1.0)
        
        # Stop linear if turning too much (turn in placeish)
        if abs(heading_error) > 0.5:
            v_cmd = 0.0
            
        return v_cmd, w_cmd

    def image_callback(self, msg):
        # 1. Preprocess Inputs
        img_input = self.preprocess_image(msg)
        if img_input is None: return

        with self.lock:
            lidar_input = self.preprocess_lidar(self.latest_scan)
            imu_input = self.preprocess_imu(self.latest_imu)
            nav_v, nav_w = self.get_nav_cmd()
            
        # 2. ANN Inference (Risk)
        inputs = {'image': img_input, 'lidar': lidar_input, 'imu': imu_input}
        try:
            risk = self.sess_ann.run(None, inputs)[0][0][0]
        except Exception:
            risk = 0.0
            
        twist = Twist()
        
        # 3. Decision Logic with Fusion
        
        # Emergency Brake
        if risk > self.risk_threshold:
            rospy.logwarn_throttle(1, f"High Risk: {risk:.2f}. Emergency Brake!")
            self.is_braking = True
            if self.brake_start_time is None:
                self.brake_start_time = rospy.Time.now()
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            
        else:
            # Check Recovery or Normal
            if self.is_braking:
                # Recovery Mode (Pure VOA Recovery)
                brake_duration = (rospy.Time.now() - self.brake_start_time).to_sec()
                try:
                    outputs = self.sess_voa.run(None, inputs)
                    recovery_out = outputs[1][0]
                    if brake_duration < self.brake_duration_short:
                        v, w = recovery_out[2], recovery_out[3] # Medium
                    else:
                        v, w = recovery_out[4], recovery_out[5] # Long
                    twist.linear.x, twist.angular.z = float(v), float(w)
                    self.is_braking = False
                    self.brake_start_time = None
                except Exception:
                    pass
            else:
                # Normal Driving -> FUSION
                if self.nav_state == "NAVIGATING":
                    try:
                        outputs = self.sess_voa.run(None, inputs)
                        policy_out = outputs[0][0] # [v, w]
                        voa_v, voa_w = float(policy_out[0]), float(policy_out[1])
                        
                        # FUSION LOGIC
                        # Factor 1: Risk (0 to threshold)
                        # Factor 2: VOA Turning Intensity (Model wants to turn)
                        
                        voa_intensity = min(abs(voa_w) / 0.8, 1.0) # Assume 0.8 is significant turn
                        
                        # Danger Factor: How much we trust VOA over Nav
                        # We map risk [0, threshold] to [0, 1]? 
                        # Or just use raw risk? Risk is usually [0, 1] output from sigmoid.
                        
                        # Let's be conservative.
                        # If Risk > 0.3, start blending heavily.
                        risk_factor = max(0.0, (risk - 0.2) / (self.risk_threshold - 0.2))
                        risk_factor = min(risk_factor, 1.0)
                        
                        # Define fusion_weight based on risk and VOA intensity
                        # We prioritize risk, but also consider if VOA wants to turn sharply
                        fusion_weight = max(risk_factor, voa_intensity * 0.3)
                        fusion_weight = min(fusion_weight, 1.0)
                        
                        if fusion_weight > 0.6:
                            self.last_avoidance_time = rospy.Time.now()
                        
                        # Anti-Oscillation Logic
                        # If we recently avoided, suppress the Global Navigator's desire to turn back
                        is_in_cooldown = (rospy.Time.now() - self.last_avoidance_time).to_sec() < self.avoidance_cooldown
                        
                        if is_in_cooldown:
                            # We are in post-avoidance. Trust VOA or go straight.
                            # Effectively, we mask out the 'nav_w' component if it contradicts or just in general
                            # to be safe, we just reduce nav_w influence to 0.
                            nav_w = 0.0
                            fusion_weight = max(fusion_weight, 0.5) # Ensure VOA has at least 50% say, or just rely on 0 nav_w
                            # Re-calculate with nav_w = 0
                            # final_w = (1 - w) * 0 + w * voa_w = w * voa_w? 
                            # If w is small (risk low), final_w -> 0 (Go Straight). This is GOOD.
                            # We want to move AWAY/PAST the obstacle, not turn back.
                        
                        # Blend Angular Velocity
                        final_w = (1.0 - fusion_weight) * nav_w + fusion_weight * voa_w
                        
                        # Blend Linear Velocity
                        # Always take the minimum for safety, but allow Nav to stop if needed (e.g. turn in place)
                        # If Nav wants to stop (v=0), we should stop (unless VOA needs to move to avoid?)
                        # VOA usually moves forward.
                        # Let's trust VOA speed if danger is high.
                        
                        if fusion_weight > 0.5:
                            final_v = voa_v # Trust VOA speed (maybe it slows down)
                        else:
                            final_v = min(nav_v, voa_v) # Safe speed
                        
                        # Stuck Detection & Recovery Override
                        if self.check_if_stuck(final_v, final_w):
                            self.is_recovering = True
                            self.recovery_start_time = rospy.Time.now()
                            rospy.logwarn("Robot Stuck! Triggering Recovery...")
                        
                        if self.is_recovering:
                            # Recovery: Reverse and Turn
                            if (rospy.Time.now() - self.recovery_start_time).to_sec() < 1.5:
                                final_v = -0.15
                                final_w = 1.0 # Rotate in place
                            else:
                                self.is_recovering = False
                                self.last_avoidance_time = rospy.Time.now() # Treat as avoidance to prevent immediate return
                            
                        twist.linear.x = float(final_v)
                        twist.angular.z = float(final_w)
                        
                        # Debug
                        # rospy.loginfo_throttle(1, f"Fusion: W_Nav={nav_w:.2f}, W_VOA={voa_w:.2f}, Risk={risk:.2f}, Wgt={fusion_weight:.2f}")
                        
                    except Exception as e:
                        rospy.logerr(f"Inference failed: {e}")
                
                elif self.nav_state == "REACHED":
                    twist.linear.x = 0.0
                    twist.angular.z = 0.0
                else:
                    # IDLE
                    pass

        self.cmd_pub.publish(twist)

if __name__ == '__main__':
    try:
        node = NavigationNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
