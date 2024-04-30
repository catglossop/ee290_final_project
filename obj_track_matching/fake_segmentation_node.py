import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image 
import rospy
import ros_numpy as rnp

class FakeSegmentationNode:

    def __init__(self):
        data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
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
            viz_combined_frames.append(Image.fromarray(viz_combined_frame))
       
        self.seg_frames = combined_frames[::15]
        self.frame_count = 0

        self.reset_sub = rospy.Subscriber('/camera/reset', Empty, self.reset_callback)
        self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        self.seg_msg = Image()

    def reset_callback(self, msg):
        self.frame_count = 0

def main():

    rospy.init_node('fake_segmentation_node', anonymous=True)
    fake_seg_node = FakeSegmentationNode()
    rate = rospy.Rate(4) # 4 Hz (for ever 15 frames published, 1 is published to the topic)

    while not rospy.is_shutdown():
        fake_seg_node.seg_msg = rnp.msgify(Image, fake_seg_node.seg_frames[fake_seg_node.frame_count%len(fake_seg_node.seg_frames)], encoding='mono8')
        fake_camera_node.seg_pub.publish(fake_seg_node.seg_msg)
        rate.sleep()

if __name__ == '__main__':
    main()


    
    