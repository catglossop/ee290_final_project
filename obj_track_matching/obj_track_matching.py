import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy



## TODO 

# 1. Improve the segmentation polygon to have improved update (Kalman Filtering)
# 2. Update with motion vector modelling as well (help to perform soft update)
data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "bird_of_paradise"
test_idx = None
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = True
SEG_PERIOD = 15

os.makedirs(f"sampled_output_{test}", exist_ok=True)

# Load in the input data 
gif_frames = []
gif_mask_frames = []
input_frames = []
seg_transformed_frames = []
opt_flow_frames = []
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
    # cv.imwrite(f"sampled_output_{test}/{test}_initial_seg_{frame_idx}.png", viz_img)
    cv.imshow("curr", viz_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# Get the features in the segmentation zone 
mask = np.zeros_like(curr_seg)
cv.fillPoly(mask, [np.flip(seg_pts, axis=2)], 255)
mask = mask.astype(np.uint8)

### LUCAS KANADE OPTICAL FLOW ###
# Get the initial features in the first image
curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)
feature_params = dict( maxCorners = 100,
 qualityLevel = 0.3,
 minDistance = 7,
 blockSize = 7 )

# Parameters for lucas kanade optical flow
lk_params = dict( winSize = (15, 15),
 maxLevel = 2,
 criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))
 
# Create some random colors
color = np.random.randint(0, 255, (100, 3))
 
# Take first frame and find corners in it
curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
curr_lk = cv.goodFeaturesToTrack(curr_gray, mask = mask, **feature_params)
 
# Create a mask image for drawing purposes
lk_mask = np.zeros_like(curr_frame)
##########################################################

while frame_idx < len(input_frames)-2: 
    # Increment the frame idx
    frame_idx += 1

    # If we have reached SEG_PERIOD
    if frame_idx % SEG_PERIOD == 0: 

        # Update the segmentation and current frame to frame_idx
        curr_seg = seg_frames[frame_idx]
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

    prev_gray = curr_gray
    prev_lk = curr_lk

    mask = np.zeros_like(curr_seg)
    cv.fillPoly(mask, [np.flip(seg_pts, axis=2)], 255)
    mask = mask.astype(np.uint8)
    gif_mask_frames.append(mask)
    # cv.imwrite(f"sampled_output_{test}/{test}_mask_{frame_idx}.png", mask)
    cv.imshow("Seg", mask)
    cv.waitKey(0)
    cv.destroyAllWindows()


    ### LUCAS KANADE OPTICAL FLOW ###
    curr_gray = cv.cvtColor(curr_frame, cv.COLOR_BGR2GRAY)
    curr_lk, st, err = cv.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_lk, mask, **lk_params)

    # Select good points
    if curr_lk is not None:
        good_new = curr_lk[st==1]
        good_old = prev_lk[st==1]

    # Compute deltas
    deltas = good_new - good_old
    avg_delta = np.mean(deltas, axis=0)
    print("Average delta: ", avg_delta)

    
    # draw the tracks
    curr_frame_lk = curr_frame.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        lk_mask = cv.line(lk_mask, (int(a), int(b)), (int(c), int(d)), color[i].tolist(), 2)
        curr_frame_lk = cv.circle(curr_frame_lk, (int(a), int(b)), 5, color[i].tolist(), -1)
        viz_img_lk = cv.add(curr_frame_lk, lk_mask)
        opt_flow_frames.append(viz_img_lk)
        cv.imshow('optical_flow', viz_img_lk)
        cv.waitKey(0)
        cv.destroyAllWindows()
    ##########################################

    # Get the orb features 
    curr_kps, curr_descs = orb.detectAndCompute(curr_frame, None)
    matches = matcher.match(prev_descs, curr_descs)
    viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
    matches = sorted(matches, key=lambda x: x.distance)

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

    # Find the set of points in the segmentation zone with good correspondences
    u_pts = []
    v_pts = []
    seg_curr_kps = ()
    seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)

    # IMPROVE THIS
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
        matrix, mask = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)
    
    except: 
        print("Homography failed")
        iio.mimsave(f'sampled_output_{test}/{test}_matching.gif', gif_frames)
        iio.mimsave(f'sampled_output_{test}/{test}_mask.gif', gif_mask_frames)
        iio.mimsave(f'sampled_output_{test}/{test}_gt.gif', seg_frames)
        # iio.mimsave(f'sampled_output_{test}/{test}_seg_transformed.gif', seg_transformed_frames)
        continue
        

    # Perform the transformation on the segmentation zone to get the new segmentation zone
    prev_seg_pts = seg_pts
    seg_pts = cv.perspectiveTransform(seg_pts.astype(np.float32), matrix)
    # seg_transformed = cv.warpPerspective(curr_seg, matrix, (curr_frame.shape[1], curr_frame.shape[0]))
    # seg_transformed = seg_transformed.astype(np.uint8)
    # seg_transformed_frames.append(seg_transformed)
    # cv.imshow("Transformed", seg_transformed)
    # cv.waitKey(0)
    # cv.destroyAllWindows()

    seg_pts = seg_pts.astype(np.int64)
    curr_kps = seg_curr_kps
    curr_descs = seg_curr_descs
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
    # cv.imwrite(f"sampled_output/{test}_matches_{frame_idx}.png", final_img)
    cv.imshow("Matches", final_img)
    cv.waitKey(0)
    cv.destroyAllWindows()

# iio.mimsave(f'sampled_output_{test}/{test}_matching.gif', gif_frames)
# iio.mimsave(f'sampled_output_{test}/{test}_mask.gif', gif_mask_frames)
# iio.mimsave(f'sampled_output_{test}/{test}_gt.gif', seg_frames)
# iio.mimsave(f'sampled_output_{test}/{test}_opt_flow.gif', opt_flow_frames)
# iio.mimsave(f'sampled_output_{test}/{test}_seg_transformed.gif', seg_transformed_frames)
print("Done")

cv.imshow('optical_flow', viz_img_lk)
cv.waitKey(0)
cv.destroyAllWindows()
breakpoint()