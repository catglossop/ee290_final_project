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
from ultralytics import YOLO
import torch
import argparse

class FakeSegmentationNode:

    def __init__(self, model_path, viz=False):

        self.image_sub = rospy.Subscriber('/camera/color/image_raw', Empty, self.image_callback)
        self.seg_pub = rospy.Publisher('/segmentation/image_raw', Image, queue_size=10)
        self.viz_pub = rospy.Publisher('/seg_viz/image_raw', Image, queue_size=10)
        self.seg_msg = Image()
        self.viz_msg = Image()
        self.bridge = CvBridge()
        torch.cuda.set_device(0)
        print("Loading model...")
        self.model = YOLO(model_path)
        print("Model loaded.")
        self.viz = viz

        self.curr_frame = None
    
    def image_callback(self, msg):
        self.curr_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        print("GOT FRAME")
        result = self.model.track(self.curr_frame)[0]
        print("GOT RESULT")
        mask = np.zeros_like(self.curr_frame[..., 0], np.int8)
        viz_mask = np.zeros_like(self.curr_frame[..., 0], np.int8)
        nseg = len(result)
        for ci, c in enumerate(res):
            contour = c.masks.xy.pop()
            contour = contour.astype(np.int32)
            contour = contour, reshape((-1, 1, 2))
            mask = cv.drawContours(mask, [contour], -1, ci+1, cv2.FILLED)
            if self.viz:
                viz_mask = cv.drawContours(viz_mask, [contour], -1, (ci+1)*(255//nseg), cv2.FILLED)
        
        self.seg_msg = self.bridge.cv2_to_imgmsg(mask, "passthrough")
        self.seg_pub.publish(self.seg_msg)
        print("RESULT PUBLISHED")
        if self.viz:
            self.viz_msg = self.bridge.cv2_to_imgmsg(self.curr_frame, "passthrough")
            self.viz_pub.publish(self.viz_msg)

def main(args):

    rospy.init_node('fake_segmentation_node', anonymous=True)
    fake_seg_node = FakeSegmentationNode(model_path=args.yolo_model, viz=args.viz)
    rospy.spin()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
                    prog='SegNode',
                    description='Segement camera stream')
    parser.add_argument('--yolo_model', type=str, default='yolov8n-seg.pt')
    parser.add_argument('--viz', type=bool, default=False)
    args = parser.parse_args()
    main(args)


    
    
