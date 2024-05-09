#!/usr/bin/env python
import rospy
import numpy as np
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

# Initialize the node
rospy.init_node('kobuki_visual_servoing')

# Publishers
cmd_vel_pub = rospy.Publisher('mobile_base/commands/velocity', Twist, queue_size=5)
debug_pub = rospy.Publisher('/debug/image', Image, queue_size=1)

# CvBridge instance
bridge = CvBridge()

# Function to stop the robot
def stop_robot():
    twist = Twist()
    twist.linear.x = 0
    twist.angular.z = 0
    cmd_vel_pub.publish(twist)

# Image callback function
def image_callback(msg):
    try:
        dtype = np.dtype("uint8") 
        n_channels = 3
        # Create a writable copy of the image data
        cv_image = np.frombuffer(msg.data, dtype=dtype).reshape(msg.height, msg.width, n_channels).copy()

        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        _, binary_mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            M = cv2.moments(largest_contour)
            if M["m00"] != 0:
                cx = int(M["m10"] / M["m00"])
                cy = int(M["m01"] / M["m00"])  # Y-coordinate of the centroid
                width = msg.width
                deviation = cx - width / 2

                twist = Twist()
                twist.angular.z = -0.001 * deviation
                twist.linear.x = -0.07
                cmd_vel_pub.publish(twist)

                # Debug visualization
                cv2.drawContours(cv_image, [largest_contour], -1, (0, 255, 0), 3)
                cv2.circle(cv_image, (cx, cy), 5, (0, 0, 255), -1)
                img_msg = bridge.cv2_to_imgmsg(cv_image, "bgr8")
                debug_pub.publish(img_msg)

                # Print centroid coordinates
                print(f"Centroid of the largest contour is at: (x={cx}, y={cy})")

            else:
                stop_robot()
        else:
            stop_robot()

    except Exception as e:
        rospy.logerr("Error in processing image for visual servoing: %s" % e)
        stop_robot()

# Subscriber
image_sub = rospy.Subscriber('/segmentation_mask_bear', Image, image_callback)

if __name__ == '__main__':
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
