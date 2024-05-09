#!/usr/bin/env python
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import CompressedImage, Image
from cv_bridge import CvBridge
from ultralytics import YOLO
import torch
import time

# Initialize the YOLO model
model = YOLO('yolov8n-seg.pt')

# ROS Node and Publisher Setup
rospy.init_node('yolo_segmentation')
pub = rospy.Publisher('/segmentation_mask', Image, queue_size=10)
bridge = CvBridge()

# Time control for publishing
last_time_published = 0
publish_interval = 0.5  # seconds

def image_callback(msg):
    global last_time_published
    current_time = time.time()
    
    # Check if it's time to publish
    if current_time - last_time_published < publish_interval:
        return  

    try:
        # Decompress the image
        np_arr = np.frombuffer(msg.data, np.uint8)
        cv_image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    except Exception as e:
        rospy.logerr("Failed to convert image: %s" % e)
        return

    # Process the image with the YOLO model
    results = model(cv_image)

    if results and results[0].masks is not None:
        height, width = cv_image.shape[:2]
        combined_mask = torch.zeros((height, width), dtype=torch.uint8)  

        masks = results[0].masks.data
        boxes = results[0].boxes.data
        clss = boxes[:, 5]  # Class IDs
        people_indices = torch.where(clss == 41)[0]  # Class ID 41 for cup

        for mask in masks[people_indices]:
            resized_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0).float(),
                                                           size=(height, width),
                                                           mode='bilinear',
                                                           align_corners=False).squeeze().int() * 255
            resized_mask = resized_mask.to('cpu')
            combined_mask = torch.maximum(combined_mask, resized_mask)

        combined_mask_np = combined_mask.cpu().numpy().astype(np.uint8)
        combined_mask_color = cv2.cvtColor(combined_mask_np, cv2.COLOR_GRAY2BGR)

        # Manually create a ROS Image message
        try:
            mask_msg = bridge.cv2_to_imgmsg(combined_mask_color, "bgr8")
            mask_msg.header.stamp = rospy.Time.now()
            mask_msg.header.frame_id = "camera"
            pub.publish(mask_msg)
            last_time_published = current_time  # Update the time we last published
        except Exception as e:
            rospy.logerr("Failed to convert processed image back to ROS message: %s" % e)

def main():
    rospy.Subscriber('/usb_cam/image_raw/compressed', CompressedImage, image_callback)
    rospy.spin()

if __name__ == '__main__':
    main()
