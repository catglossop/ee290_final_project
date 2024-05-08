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
        data_path = "/home/ee290"
        input_path = os.path.join(data_path, "extracted_frames")
        input_test_path = input_path
        input_frames = []

        for img in sorted(os.listdir(os.path.join(input_test_path))):
            frame = cv.imread(os.path.join(input_test_path, img))
            frame = cv.resize(frame, (360, 640))
            input_frames.append(frame)

        gt_path = os.path.join(data_path, "segmented_video")
        gt_test_path = gt_path

        combined_frames = []
        viz_combined_frames = []
        for img in sorted(os.listdir(os.path.join(gt_test_path))):
            frame = cv.imread(os.path.join(gt_test_path, img))[...,0]
            frame = cv.resize(frame, (360, 640))
            combined_frames.append(frame)
            viz_combined_frames.append((frame*(255//np.max(frame))))

        self.input_frames = input_frames
        self.frame_count = 0
        self.input_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/camera/reset', Empty, queue_size=10)
        self.image_msg = Image()
        self.bridge = CvBridge()

        self.periods = [10, 20, 30, 40, 50, 60, 70]
        self.seg_period = self.periods.pop(0)
        self.gt_frames = combined_frames
        self.seg_frames = combined_frames[::self.seg_period]
        self.org_viz_combined_frames = viz_combined_frames
        self.viz_combined_frames = viz_combined_frames[::self.seg_period]
        self.frame_count = 0
        self.seg_count = 0

        self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        self.viz_pub = rospy.Publisher('/seg_viz/image_raw', Image, queue_size=10)
        self.gt_pub = rospy.Publisher('/ground_truth/image_raw', Image, queue_size=10)
        self.seg_msg = Image()
        self.viz_msg = Image() 
        self.iter_count = 0


def main():

    rospy.init_node('fake_camera_node', anonymous=True)
    fake_camera_node = FakeCameraNode()
    rate = rospy.Rate(60) # 60Hz


    while not rospy.is_shutdown():
        if fake_camera_node.frame_count % len(fake_camera_node.input_frames) == 0 and fake_camera_node.frame_count != 0:
            fake_camera_node.reset_pub.publish(Empty())
            fake_camera_node.frame_count = 0
            fake_camera_node.seg_count = 0
            fake_camera_node.iter_count += 1
            if fake_camera_node.iter_count % 5 == 0:
                fake_camera_node.seg_period = fake_camera_node.periods.pop(0)
                fake_camera_node.seg_frames =fake_camera_node.gt_frames[::fake_camera_node.seg_period]
                fake_camera_node.viz_combined_frames = fake_camera_node.org_viz_combined_frames[::fake_camera_node.seg_period]

        fake_camera_node.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.input_frames[fake_camera_node.frame_count%len(fake_camera_node.input_frames)], "passthrough")
        fake_camera_node.input_pub.publish(fake_camera_node.image_msg)
        fake_camera_node.gt_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.gt_frames[fake_camera_node.frame_count%len(fake_camera_node.input_frames)], "passthrough")
        fake_camera_node.gt_pub.publish(fake_camera_node.gt_msg)
        fake_camera_node.frame_count += 1
        if fake_camera_node.frame_count % fake_camera_node.seg_period == 0:
            fake_camera_node.seg_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.seg_frames[fake_camera_node.seg_count], "passthrough")
            fake_camera_node.viz_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.viz_combined_frames[fake_camera_node.seg_count], "passthrough")
            fake_camera_node.seg_pub.publish(fake_camera_node.seg_msg)
            fake_camera_node.viz_pub.publish(fake_camera_node.viz_msg)
            fake_camera_node.seg_count += 1

        rate.sleep()
    

if __name__ == '__main__':
    main()


    
    
