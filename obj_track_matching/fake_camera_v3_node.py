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


class FakeCameraV3Node:

    def __init__(self):

        # GET TEST DATA
        video_path = "/home/roboslice/eval_vid_0.mp4"
        test_label = 0

        self.input_frames = input_frames
        self.frame_count = 0
        self.input_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/camera/reset', Empty, queue_size=10)
        self.image_msg = Image()
        self.bridge = CvBridge()

        self.cap = cv.VideoCapture(video_path)

        # self.periods = [10, 20, 30, 40, 50, 60, 70]
        # self.seg_period = self.periods.pop(0)
        # self.gt_frames = combined_frames
        # self.seg_frames = combined_frames[::self.seg_period]
        # self.org_viz_combined_frames = viz_combined_frames
        # self.viz_combined_frames = viz_combined_frames[::self.seg_period]
        # self.frame_count = 0
        # self.seg_count = 0

        # self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        # self.viz_pub = rospy.Publisher('/seg_viz/image_raw', Image, queue_size=10)
        # self.gt_pub = rospy.Publisher('/ground_truth/image_raw', Image, queue_size=10)
        # self.seg_msg = Image()
        # self.viz_msg = Image() 
        self.iter_count = 0


def main():

    rospy.init_node('fake_camera_node', anonymous=True)
    fake_camera_node = FakeCameraNode()
    rate = rospy.Rate(60) # 60Hz


    while not rospy.is_shutdown():

        ret, frame = fake_camera_node.cap.read()
        fake_camera_msg.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(frame, "passthrough")
        fake_camera_node.input_pub.publish(fake_camera_node.image_msg)

        rate.sleep()
    

if __name__ == '__main__':
    main()


    
    
