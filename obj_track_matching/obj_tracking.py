import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time


data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "frog"
test_idx = None
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = True
SEG_PERIOD = 20

os.makedirs(f"output/sampled_output_{test}", exist_ok=True)

# Load in the input data 
gif_frames = []
gif_mask_frames = []
input_frames = []
seg_transformed_frames = []
opt_flow_frames = []
IOU_est = []
IOU_gt = []
loop_time = []
for img in sorted(os.listdir(os.path.join(input_test_path))):
    frame = cv.imread(os.path.join(input_test_path, img))
    input_frames.append(frame)

seg_frames = []
if test_idx is not None:
    gt_test_path = os.path.join(gt_test_path, test_idx)
for img in sorted(os.listdir(os.path.join(gt_test_path))):
    frame = iio.imread(os.path.join(gt_test_path,img))
    seg_frames.append(frame[..., 0])


# Initialize orb feature detector and matcher
orb = cv.ORB_create() 
matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

# Get the current seg and input frames
frame_idx = 0
curr_seg = seg_frames[frame_idx]
seg_transformed = curr_seg
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
    viz_img = cv.polylines(viz_img, [np.flip(seg_pts, axis=2)], True, (255,255,255), 1)
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

while frame_idx < len(input_frames)-2: 
    # Increment the frame idx
    frame_idx += 1

    start = time.time()

    # If we have reached SEG_PERIOD
    if frame_idx % SEG_PERIOD == 0: 

        # Update the segmentation and current frame to frame_idx
        curr_seg = seg_frames[frame_idx]
        seg_transformed = curr_seg
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
            viz_img = cv.polylines(viz_img, [np.flip(seg_pts, axis=2)], True, (255,255,255), 1)
            viz_img = cv.circle(viz_img, (seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 0, 255), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 0), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(255, 0, 0), thickness=-1)
            viz_img = cv.circle(viz_img, (seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 255), thickness=-1)
            # cv.imwrite(f"sampled_output_{test}/{test}_updated_seg_{frame_idx}.png", viz_img)
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
    if VISUALIZE:
        gif_mask_frames.append(mask)
        # cv.imwrite(f"sampled_output_{test}/{test}_mask_{frame_idx}.png", mask)
        cv.imshow("Seg", mask)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Get the orb features 
    curr_kps, curr_descs = orb.detectAndCompute(curr_frame, None)
    matches = matcher.match(prev_descs, curr_descs)
    viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
    matches = sorted(matches, key=lambda x: x.distance)

    if VISUALIZE:
        final_img = cv.drawMatches(prev_frame, prev_kps, curr_frame, curr_kps, viz_matches, None) 
        final_img = cv.polylines(final_img, [np.flip(init_seg_pts, axis=2)], True, (255,255,255), 1)
        final_img = cv.circle(final_img, (init_seg_pts[0,0,1], init_seg_pts[0,0,0]), 5, color=(0, 0, 255), thickness=-1)
        final_img = cv.circle(final_img, (init_seg_pts[1,0,1], init_seg_pts[1,0,0]), 5, color=(0, 0, 255), thickness=-1)
        final_img = cv.circle(final_img, (init_seg_pts[2,0,1], init_seg_pts[2,0,0]), 5, color=(0, 0, 255), thickness=-1)
        final_img = cv.circle(final_img, (init_seg_pts[3,0,1], init_seg_pts[3,0,0]), 5, color=(0, 0, 255), thickness=-1)

        final_img = cv.circle(final_img, (seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 255, 255), thickness=-1)
        final_img = cv.circle(final_img, (seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 255), thickness=-1)
        final_img = cv.circle(final_img, (seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(0, 255, 255), thickness=-1)
        final_img = cv.circle(final_img, (seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 255), thickness=-1)

        cv.imshow("Matches", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Find the set of points in the segmentation zone with good correspondences
    u_pts = []
    v_pts = []
    seg_curr_kps = ()
    seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)

    for match in matches:
        if curr_kps[match.trainIdx].pt[0] < seg_pts[3,0,1] or curr_kps[match.trainIdx].pt[0] > seg_pts[1,0,1] or curr_kps[match.trainIdx].pt[1] < seg_pts[0,0,0] or curr_kps[match.trainIdx].pt[1] > seg_pts[2,0,0]:
            viz_img = curr_frame.copy()
            viz_img = cv.polylines(viz_img, [np.flip(seg_pts, axis=2)], True, (255,255,255), 1)
            pt = (curr_kps[match.trainIdx].pt[0], curr_kps[match.trainIdx].pt[1])
            viz_img = cv.circle(viz_img, (int(curr_kps[match.trainIdx].pt[0]), int(curr_kps[match.trainIdx].pt[1])), 5, color=(0, 0, 255), thickness=-1)
            continue
        else:
            u_pts.append(prev_kps[match.queryIdx].pt)
            v_pts.append(curr_kps[match.trainIdx].pt)
            seg_curr_kps = seg_curr_kps + (curr_kps[match.trainIdx],)
            seg_curr_descs = np.vstack((seg_curr_descs, curr_descs[match.trainIdx]))

    u_pts = np.flip(np.float32(u_pts).reshape(-1, 1, 2), axis=2)
    v_pts = np.flip(np.float32(v_pts).reshape(-1, 1, 2), axis=2)
    
    # Get the homography defined by this set of points
    try: 
        matrix, _ = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)
    
    except: 
        print("Homography failed")
        continue
        

    # Perform the transformation on the segmentation zone to get the new segmentation zone
    prev_seg_pts = seg_pts
    seg_pts = cv.perspectiveTransform(seg_pts.astype(np.float32), matrix)

    # Measure delta between the previous and current seg pts 
    delta = seg_pts - prev_seg_pts

    # Get IOU weight
    prev_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(prev_seg_pts.astype(np.int64), axis=2)], 255)
    curr_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(seg_pts.astype(np.int64), axis=2)], 255)
    iou = np.sum(np.logical_and(prev_mask, curr_mask)) / np.sum(np.logical_or(prev_mask, curr_mask))
    print("IOU: ", iou)
    if iou < 0.5:
        seg_pts = prev_seg_pts
        continue
    prev_seg_pts = prev_seg_pts + iou*delta

    # Get IOU with GT segmentation
    gt_mask = seg_frames[frame_idx]
    iou_gt = np.sum(np.logical_and(gt_mask, curr_mask)) / np.sum(np.logical_or(gt_mask, curr_mask))
    print("IOU GT: ", iou_gt)

    IOU_est.append(iou)
    IOU_gt.append(iou_gt)

    seg_pts = seg_pts.astype(np.int64)
    curr_kps = seg_curr_kps
    curr_descs = seg_curr_descs

    end = time.time()
    loop_time.append((end-start)*1000)
    print("Time: ", (end-start)*1000, "ms")

    if VISUALIZE:
        shifted_seg_pts = seg_pts.copy()
        shifted_seg_pts[0,0,1] += curr_frame.shape[1]
        shifted_seg_pts[1,0,1] += curr_frame.shape[1]
        shifted_seg_pts[2,0,1] += curr_frame.shape[1]
        shifted_seg_pts[3,0,1] += curr_frame.shape[1]
        final_img = cv.polylines(final_img, [np.flip(shifted_seg_pts, axis=1)], True, (255,255,255), 1)
        final_img = cv.circle(final_img, (curr_frame.shape[1] + shifted_seg_pts[0,0,1], seg_pts[0,0,0]), 5, color=(0, 255, 0), thickness=-1)
        final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[1,0,1], seg_pts[1,0,0]), 5, color=(0, 255, 0), thickness=-1)
        final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[2,0,1], seg_pts[2,0,0]), 5, color=(0, 255, 0), thickness=-1)
        final_img = cv.circle(final_img, (curr_frame.shape[1] + seg_pts[3,0,1], seg_pts[3,0,0]), 5, color=(0, 255, 0), thickness=-1)

        gif_frames.append(final_img)

        # Show the final image 
        cv.imshow("Matches", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

fig, ax = plt.subplots(1,2)
ax[0].plot(IOU_est)
ax[0].plot(IOU_gt)
ax[1].plot(loop_time)
ax[0].legend(["Estimated IOU", "GT IOU", "loop time"])
ax[0].set_xlabel("Frame")
ax[0].set_ylabel("IOU")
ax[1].set_xlabel("Frame")
ax[1].set_ylabel("Time (ms)")
ax[0].set_title("IOU vs Frame")
ax[1].set_title("Loop Time vs Frame")
plt.show()

if len(gif_frames) != 0:
    print("Saving gifs")
    iio.mimsave(f'output/sampled_output_{test}/{test}_matching.gif', gif_frames)
    iio.mimsave(f'output/sampled_output_{test}/{test}_mask.gif', gif_mask_frames)
    iio.mimsave(f'output/sampled_output_{test}/{test}_gt.gif', seg_frames)

print("Done")
