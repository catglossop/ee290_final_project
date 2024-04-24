import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy

data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "bird_of_paradise"
test_idx = None
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = True
SEG_PERIOD = 15

# Load in the input data 

input_frames = []
for img in sorted(os.listdir(os.path.join(input_test_path))):
    frame = cv.imread(os.path.join(input_test_path, img))
    input_frames.append(frame)

seg_frames = []
if test_idx is not None:
    gt_test_path = os.path.join(gt_test_path, test_idx)
for img in sorted(os.listdir(os.path.join(gt_test_path))):
    frame = iio.imread(os.path.join(gt_test_path,img))
    seg_frames.append(frame)


# Initialize orb feature detector and matcher
orb = cv.ORB_create()
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Get the current seg and input frames
frame_idx = 0
curr_seg = seg_frames[frame_idx][..., 0]
curr_frame = input_frames[frame_idx]


# Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
arg_mask = np.argwhere(curr_seg == 255)
Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

# Turn the points into an array 
seg_pts = np.array([Px_min, Py_max, Px_max, Py_min])
seg_pts = seg_pts.reshape(-1, 1, 2)
init_seg_pts = seg_pts

# Visualize
if VISUALIZE:
    viz_img = curr_frame.copy()
    viz_img = cv.circle(viz_img, (seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 0, 255), thickness=-1)
    viz_img = cv.circle(viz_img, (seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 0), thickness=-1)
    viz_img = cv.circle(viz_img, (seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(255, 0, 0), thickness=-1)
    viz_img = cv.circle(viz_img, (seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 255), thickness=-1)
    cv.imshow("curr", viz_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Get the features in the segmentation zone 
mask = np.zeros_like(curr_seg)
cv.fillPoly(mask, [np.flip(seg_pts, axis=2)], 255)
mask = mask.astype(np.uint8)

# Get the initial features in the first image
curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)

while frame_idx < len(input_frames)-2: 
    # Increment the frame idx
    frame_idx += 1

    # If we have reached SEG_PERIOD
    if frame_idx % SEG_PERIOD == 0: 

        # Update the segmentation and current frame to frame_idx
        curr_seg = seg_frames[frame_idx][..., 0]
        curr_frame = input_frames[frame_idx]
        frame_idx += 1
        
        # Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
        arg_mask = np.argwhere(curr_seg == 255)
        Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
        Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
        Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
        Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

        seg_pts = np.array([Px_min, Py_max, Px_max, Py_min])
        seg_pts = seg_pts.reshape(-1, 1, 2)
        init_seg_pts = seg_pts

        # Visualize
        if VISUALIZE:
            viz_img = curr_frame.copy()
            viz_img = cv.circle(viz_img, (seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 0, 255), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 0), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(255, 0, 0), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 255), thickness=-1)
            cv.imshow("curr", viz_img)
            cv.waitKey(0)
            cv.destroyAllWindows()


        # Get the features in the segmentation zone 
        mask = np.zeros_like(curr_seg)
        cv.fillPoly(mask, [np.flip(seg_pts, axis=2)], 255)
        mask = mask.astype(np.uint8)
        curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)

    # Get the next frame 
    prev_frame = curr_frame
    prev_kps = curr_kps
    prev_descs = curr_descs
    curr_frame = input_frames[frame_idx]

    mask = np.zeros_like(curr_seg)
    cv.fillPoly(mask, [np.flip(seg_pts, axis=2)], 255)
    mask = mask.astype(np.uint8)
    cv.imshow("Seg", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()

    # Get the orb features 
    curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)
    matches = matcher.match(prev_descs, curr_descs)
    matches = sorted(matches, key=lambda x: x.distance)
    # matches = matches
    final_img = cv.drawMatches(prev_frame, prev_kps, curr_frame, curr_kps, matches, None) 
    final_img = cv.circle(final_img, (init_seg_pts[0,0,1], init_seg_pts[0,0,0]), 5, color=(0, 0, 255), thickness=-1)
    final_img = cv.circle(final_img, (init_seg_pts[1,0,1], init_seg_pts[1,0,0]), 5, color=(0, 0, 255), thickness=-1)
    final_img = cv.circle(final_img, (init_seg_pts[2,0,1], init_seg_pts[2,0,0]), 5, color=(0, 0, 255), thickness=-1)
    final_img = cv.circle(final_img, (init_seg_pts[3,0,1], init_seg_pts[3,0,0]), 5, color=(0, 0, 255), thickness=-1)

    final_img = cv.circle(final_img, (seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 255, 255), thickness=-1)
    final_img = cv.circle(final_img, (seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 255), thickness=-1)
    final_img = cv.circle(final_img, (seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(0, 255, 255), thickness=-1)
    final_img = cv.circle(final_img, (seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 255), thickness=-1)

    # Find the set of points in the segmentation zone with good correspondences
    u_pts = []
    v_pts = []
    seg_curr_kps = ()
    seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)

    # IMPROVE THIS
    for match in matches:
        # if curr_kps[match.trainIdx].pt[0] < seg_pts[3,0,1] or curr_kps[match.trainIdx].pt[0] > seg_pts[1,0,1] or curr_kps[match.trainIdx].pt[1] < seg_pts[0,0,0] or curr_kps[match.trainIdx].pt[1] > seg_pts[2,0,1]:
        #     print(curr_kps[match.trainIdx].pt)
        #     print("X min: ", seg_pts[0,0,0])
        #     print("X max: ", seg_pts[2,0,0])
        #     print("Y min: ", seg_pts[3,0,1])
        #     print("Y max: ", seg_pts[1,0,1])
        #     # breakpoint()
        #     continue
        u_pts.append(prev_kps[match.queryIdx].pt)
        v_pts.append(curr_kps[match.trainIdx].pt)
        seg_curr_kps = seg_curr_kps + (curr_kps[match.trainIdx],)
        seg_curr_descs = np.vstack((seg_curr_descs, curr_descs[match.trainIdx]))

    u_pts = np.float32(u_pts).reshape(-1, 1, 2)
    v_pts = np.float32(v_pts).reshape(-1, 1, 2)
    
    # Get the homography defined by this set of points
    print("Init seg pts: ", init_seg_pts)
    print("seg pts: ", seg_pts)
    matrix, mask = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)

    # Perform the transformation on the segmentation zone to get the new segmentation zone
    prev_seg_pts = seg_pts
    seg_pts = cv.perspectiveTransform(seg_pts.astype(np.float32), matrix)
    seg_pts = seg_pts.astype(np.int64)
    print("Shape of curr desc before: ", curr_descs.shape)
    print("Shape of curr desc after: ", seg_curr_descs.shape)
    print("Num matches: ", len(matches))
    curr_kps = seg_curr_kps
    curr_descs = seg_curr_descs
    final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 255, 0), thickness=-1)
    final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 0), thickness=-1)
    final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(0, 255, 0), thickness=-1)
    final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 0), thickness=-1)

    # Show the final image 
    cv.imshow("Matches", final_img)
    cv.waitKey(0)
    cv.destroyAllWindows()


print("Done")














