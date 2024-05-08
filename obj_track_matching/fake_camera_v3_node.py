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

    def __init__(self, video_path, gt_path, test_label, fps):

        # GET TEST DATA
        self.video_path = video_path
        self.gt_path =  gt_path
        print(self.video_path)
        self.test_label = test_label
        self.fps = fps

        # self.input_frames = input_frames
        # self.frame_count = 0
        self.input_pub = rospy.Publisher('/yolo/camera/color/image_raw', Image, queue_size=10)
        self.gt_pub = rospy.Publisher('/yolo/ground_truth/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/yolo/camera/reset', Empty, queue_size=10)
        self.seg_sub = rospy.Subscriber('/yolo/segmentation/image_raw', Image, self.seg_callback)
        self.first_seg_processed = False
        self.image_msg = Image()
        self.gt_msg = Image()
        self.bridge = CvBridge()

        self.cap = cv.VideoCapture(video_path)
        self.cap_gt = cv.VideoCapture(gt_path)
        self.iter_count = 0
    
    def seg_callback(self, msg):
        print("First seg processed!")
        self.first_seg_processed = True



def main(args):
    print(args)
    rospy.init_node('fake_camera_v3_node', anonymous=True)
    fake_camera_node = FakeCameraV3Node(args.video_path, args.gt_path, args.test_label, args.fps)
    rate = rospy.Rate(fake_camera_node.fps) # 60Hz


    while not rospy.is_shutdown():

        if fake_camera_node.first_seg_processed or fake_camera_node.first_frame: 
            if fake_camera_node.first_frame: 
                fake_camera_node.first_frame = False
            ret, frame = fake_camera_node.cap.read()
            frame = np.array(frame)
            ret, gt_frame = fake_camera_node.cap_gt.read()
            if frame is None or gt_frame is None:
                fake_camera_node.reset_pub.publish(Empty())
            gt_frame = np.array(gt_frame)
            print("frame publishing...")
            fake_camera_node.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(frame, "passthrough")
            fake_camera_node.gt_msg = fake_camera_node.bridge.cv2_to_imgmsg(gt_frame, "passthrough")
            fake_camera_node.input_pub.publish(fake_camera_node.image_msg)
            fake_camera_node.gt_pub.publish(fake_camera_node.gt_msg)

        rate.sleep()
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Fake Camera Node')
    parser.add_argument('--video_path', type=str, default='/home/roboslice/ee290/eval_vid_1.avi')
    parser.add_argument('--gt_path', type=str, default='/home/roboslice/ee290/eval_vid_1_gt.avi')
    parser.add_argument('--test_label', type=int, default=0)
    parser.add_argument('--fps', type=int, default=60)
    args = parser.parse_args()
    main(args)


    
    
