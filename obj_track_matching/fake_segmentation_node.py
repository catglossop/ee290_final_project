import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
from sensor_msgs.msg import Image 
from std_msgs.msg import Empty
import rospy
from cv_bridge import CvBridge # INSTALL ON PI

class FakeSegmentationNode:

    def __init__(self):
        data_path = "/home/proj206a/data/SegTrackv2"
        gt_path = os.path.join(data_path, "GroundTruth")
        test = "frog"
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
       
        self.seg_frames = combined_frames[::15]
        self.viz_combined_frames = viz_combined_frames[::15]
        self.frame_count = 0

        self.reset_sub = rospy.Subscriber('/camera/reset', Empty, self.reset_callback)
        self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        self.viz_pub = rospy.Publisher('/seg_viz/image_raw', Image, queue_size=10)
        self.seg_msg = Image()
        self.viz_msg = Image()
        self.bridge = CvBridge()

    def reset_callback(self, msg):
        self.frame_count = 0

def main():

    rospy.init_node('fake_segmentation_node', anonymous=True)
    fake_seg_node = FakeSegmentationNode()
    rate = rospy.Rate(4) # 4 Hz (for ever 15 frames published, 1 is published to the topic)

    while not rospy.is_shutdown():
        print("Sending seg image")
        fake_seg_node.seg_msg = fake_seg_node.bridge.cv2_to_imgmsg(fake_seg_node.seg_frames[fake_seg_node.frame_count%len(fake_seg_node.seg_frames)], "passthrough")
        fake_seg_node.viz_msg = fake_seg_node.bridge.cv2_to_imgmsg(fake_seg_node.viz_combined_frames[fake_seg_node.frame_count%len(fake_seg_node.viz_combined_frames)], "passthrough")
        fake_seg_node.seg_pub.publish(fake_seg_node.seg_msg)
        fake_seg_node.viz_pub.publish(fake_seg_node.viz_msg)
        fake_seg_node.frame_count += 1
        rate.sleep()

if __name__ == '__main__':
    main()


    
    
