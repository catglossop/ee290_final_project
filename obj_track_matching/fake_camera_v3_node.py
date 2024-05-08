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
import argparse


class FakeCameraV3Node:

    def __init__(self, video_path, test_label, fps):

        # GET TEST DATA
        self.video_path = video_path
        print(self.video_path)
        self.test_label = test_label
        self.fps = fps

        # self.input_frames = input_frames
        # self.frame_count = 0
        self.input_pub = rospy.Publisher('/yolo/camera/color/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/yolo/camera/reset', Empty, queue_size=10)
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


def main(args):
    print(args)
    rospy.init_node('fake_camera_v3_node', anonymous=True)
    fake_camera_node = FakeCameraV3Node(args.video_path, args.test_label, args.fps)
    rate = rospy.Rate(fake_camera_node.fps) # 60Hz


    while not rospy.is_shutdown():

        ret, frame = fake_camera_node.cap.read()
        frame = np.array(frame)
        print("frame publishing...")
        fake_camera_node.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(frame, "passthrough")
        fake_camera_node.input_pub.publish(fake_camera_node.image_msg)

        rate.sleep()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake Camera Node')
    args = parser.parse_args()
    parser.add_argument('--video_path', type=str, default='/home/roboslice/ee290/eval_vid_1.avi')
    parser.add_argument('--test_label', type=int, default=0)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()
    main(args)


    
    
