#!/usr/bin/env python3
import rospy
import sys
from geometry_msgs.msg import PoseStamped
import tf.transformations

def send_goal(x, y, theta):
    rospy.init_node('send_goal_client', anonymous=True)
    pub = rospy.Publisher('/move_base_simple/goal', PoseStamped, queue_size=1)
    
    # Wait for publisher to connect
    rospy.sleep(1)
    
    goal = PoseStamped()
    goal.header.frame_id = "map" # Or "odom" depending on your setup
    goal.header.stamp = rospy.Time.now()
    
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.position.z = 0.0
    
    # Convert theta (yaw) to quaternion
    q = tf.transformations.quaternion_from_euler(0, 0, theta)
    goal.pose.orientation.x = q[0]
    goal.pose.orientation.y = q[1]
    goal.pose.orientation.z = q[2]
    goal.pose.orientation.w = q[3]
    
    rospy.loginfo(f"Publishing Goal: x={x}, y={y}, theta={theta}")
    pub.publish(goal)
    rospy.sleep(1) # Ensure message is sent
    rospy.loginfo("Goal sent!")

if __name__ == '__main__':
    if len(sys.argv) < 3:
        print("Usage: send_goal.py <x> <y> [theta]")
        sys.exit(1)
        
    x = float(sys.argv[1])
    y = float(sys.argv[2])
    theta = float(sys.argv[3]) if len(sys.argv) > 3 else 0.0
    
    try:
        send_goal(x, y, theta)
    except rospy.ROSInterruptException:
        pass
