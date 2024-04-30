import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time 
import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from cv_bridge import CvBridge # INSTALL ON PI


class FakeCameraNode:

    def __init__(self):

        # GET TEST DATA
        data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
        input_path = os.path.join(data_path, "JPEGImages")
        test = "frog"
        input_test_path = os.path.join(input_path, test)
        input_frames = []

        for img in sorted(os.listdir(os.path.join(input_test_path))):
            frame = cv.imread(os.path.join(input_test_path, img))
            input_frames.append(frame)

        self.input_frames = input_frames
        self.frame_count = 0
        self.input_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/camera/reset', Empty, queue_size=10)
        self.image_msg = Image()
        self.bridge = CvBridge()

def main():

    rospy.init_node('fake_camera_node', anonymous=True)
    fake_camera_node = FakeCameraNode()
    rate = rospy.Rate(60) # 60Hz


    while not rospy.is_shutdown():
        if fake_camera_node.frame_count % len(fake_camera_node.input_frames) == 0:
            fake_camera_node.reset_pub.publish(Empty())
        fake_camera_node.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.input_frames[fake_camera_node.frame_count%len(fake_camera_node.input_frames)], "passthrough")
        fake_camera_node.input_pub.publish(fake_camera_node.image_msg)
        rate.sleep()
    

if __name__ == '__main__':
    main()


    
    