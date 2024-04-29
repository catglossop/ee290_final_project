import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image 
import rospy


data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "frog"
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = True
LK = False
SEG_PERIOD = 15
DEBUG = False


for img in sorted(os.listdir(os.path.join(input_test_path))):
    frame = cv.imread(os.path.join(input_test_path, img))
    input_frames.append(frame)

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

num_segs = len(seg_frames.keys())
seg_frames = combined_frames
seg_color = np.random.randint(0, 255, (num_segs*2, 3))

class FakeCameraNode:

    def __init__(self):


        self.seg_frames = seg_frames
        self.input_frames = input_frames
        self.num_segs = num_segs
        self.frame_count = 0

        self.input_pub = rospy.Publisher('/fake_camera/input', Image, queue_size=10)
        self.seg_pub = rospy.Publisher('/fake_camera/seg', Image, queue_size=10)

    


def main():

    rospy.init_node('fake_camera_node', anonymous=True)
    fake_camera_node = FakeCameraNode()
    rate = rospy.Rate(30) # 30hz

    while not rospy.is_shutdown():
        input_frame = Image.fromarray(frame)
        seg_frame = Image.fromarray(fake_camera_node.seg_frames[idx])
        fake_camera_node.input_pub.publish(input_frame)
        fake_camera_node.seg_pub.publish(seg_frame)
        rate.sleep()


    
    