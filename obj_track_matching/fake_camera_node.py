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
        data_path = "/home/proj206a/data/SegTrackv2"
        input_path = os.path.join(data_path, "JPEGImages")
        test = "frog"
        input_test_path = os.path.join(input_path, test)
        input_frames = []

        for img in sorted(os.listdir(os.path.join(input_test_path))):
            frame = cv.imread(os.path.join(input_test_path, img))
            input_frames.append(frame)

        gt_path = os.path.join(data_path, "GroundTruth")
        gt_test_path = os.path.join(gt_path, test)

        seg_frames = {1: []}
        for sub in sorted(os.listdir(os.path.join(gt_test_path))):
            sub_path = os.path.join(gt_test_path, sub)
            if os.path.isdir(sub_path):
                seg_frames[int(sub)] = []
                for img in sorted(os.listdir(sub_path)):
                    frame = iio.imread(os.path.join(sub_path,img))
                    seg_frames[int(sub)].append(frame[..., 0])
            else:
                frame = iio.imread(sub_path)
                seg_frames[1].append(frame[..., 0])

        # Combine together frames
        combined_frames = []
        viz_combined_frames = []
        for idx in enumerate(seg_frames[1]):
            combined_frame = np.zeros_like(seg_frames[1][idx[0]])
            viz_combined_frame = np.zeros_like(seg_frames[1][idx[0]])
            for key in seg_frames.keys():
                combined_frame[seg_frames[key][idx[0]] == 255] = key
                viz_combined_frame[seg_frames[key][idx[0]] == 255] = key*(255//len(seg_frames.keys()))
            combined_frames.append(combined_frame)
            viz_combined_frames.append(viz_combined_frame)

        self.input_frames = input_frames
        self.frame_count = 0
        self.input_pub = rospy.Publisher('/camera/color/image_raw', Image, queue_size=10)
        self.reset_pub = rospy.Publisher('/camera/reset', Empty, queue_size=10)
        self.image_msg = Image()
        self.bridge = CvBridge()

        self.seg_period = 30
        self.seg_frames = combined_frames[::self.seg_period]
        self.viz_combined_frames = viz_combined_frames[::self.seg_period]
        self.frame_count = 0
        self.seg_count = 0

        self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        self.viz_pub = rospy.Publisher('/seg_viz/image_raw', Image, queue_size=10)
        self.gt_pub = rospy.Publisher('/ground_truth/image_raw', Image, queue_size=10)
        self.seg_msg = Image()
        self.viz_msg = Image()


def main():

    rospy.init_node('fake_camera_node', anonymous=True)
    fake_camera_node = FakeCameraNode()
    rate = rospy.Rate(60) # 60Hz


    while not rospy.is_shutdown():
        if fake_camera_node.frame_count % len(fake_camera_node.input_frames) == 0:
            fake_camera_node.reset_pub.publish(Empty())
            fake_camera_node.frame_count = 0
            fake_camera_node.seg_count = 0
        
        fake_camera_node.image_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.input_frames[fake_camera_node.frame_count%len(fake_camera_node.input_frames)], "passthrough")
        fake_camera_node.input_pub.publish(fake_camera_node.image_msg)
        fake_camera_node.gt_msg = fake_camera_node.bridge.cv2_to_imgmsg(fake_camera_node.seg_frames[fake_camera_node.frame_count%len(fake_camera_node.input_frames)], "passthrough")
        fake_camera_node.gt_pub.publish()
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


    
    
