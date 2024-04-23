import os 
import cv2
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt



data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "bird_of_paradise"
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = False

# Load in the input data 

input_frames = []
for img in sorted(os.listdir(input_test_path)):
    frame = iio.imread(os.path.join(input_test_path, img))
    input_frames.append(frame)

seg_frames = []
for img in sorted(os.listdir(gt_test_path)):
    frame = iio.imread(os.path.join(gt_test_path, img))
    seg_frames.append(frame)


# Initialize with the first segmented image 

while not len(seg_frames) == 0: 

    init_seg = seg_frames.pop(0)[..., 0]
    curr_seg = init_seg
    curr_frame = input_frames.pop(0)

    # Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
    arg_mask = np.argwhere(curr_seg == 255)

    Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
    Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
    Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
    Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

    # Visualize

    if VISUALIZE:
        plt.imshow(init_seg)
        plt.plot(Px_max[1], Px_max[0], 'ro')
        plt.plot(Px_min[1], Px_min[0], 'ro')
        plt.plot(Py_max[1], Py_max[0], 'ro')
        plt.plot(Py_min[1], Py_min[0], 'ro')
        plt.show()
    
    # Use current frame to get features in segmentation zone 
    # Get features in the next frame (rapidly --> fast feature matching)
    # Get motion/transformation estimate from previous frame to current frame
    # Error est 
        # if big error --> re segment 
        # if small error --> continue
    # Otherwise continue to loop through the frames 












