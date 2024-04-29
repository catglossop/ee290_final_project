import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image 
from multiprocessing import Pool

## SETUP FOR TESTING
data_path = "/Users/catherineglossop/ee290_final_project/data/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
test = "penguin"
input_test_path = os.path.join(input_path, test)
gt_test_path = os.path.join(gt_path, test)
VISUALIZE = True
LK = False
SEG_PERIOD = 15
DEBUG = False
os.makedirs(f"output/sampled_output_{test}", exist_ok=True)

# Load in the input data 
gif_frames = []
gif_mask_frames = []
input_frames = []
opt_flow_frames = []
IOU_est = []
IOU_gt = []
loop_time = []

for img in sorted(os.listdir(os.path.join(input_test_path))):
    frame = cv.imread(os.path.join(input_test_path, img))
    input_frames.append(frame)

def get_process_dirs(num_processes=4):
    process_segs = []
    dirs = sorted(os.listdir(os.path.join(gt_test_path)))
    breakpoint()
    if os.path.isdir(os.path.join(gt_test_path, dirs[0])):
        num_segs = len(dirs)
    else:
        num_segs = 1
        return [os.path.join(gt_test_path)]
    segs_per_process = num_segs // num_processes

    process_segs = [[os.path.join(gt_test_path, dirs[j]) for subpath in dirs[i*segs_per_process:(i+1)*segs_per_process]] for i in range(num_processes)]
    return process_segs

def propogate_segmentations(process_dirs, input_frames):

    # Get sub segmentations
    seg_frames = {1: []}
    for path in sorted(process_dirs):
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

    # Initialize orb feature detector and matcher
    orb = cv.ORB_create() 
    matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Get the current seg and input frames
    frame_idx = 0
    curr_seg = seg_frames[frame_idx]
    curr_frame = input_frames[frame_idx]

    # Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
    seg_pts = {}
    for nseg in range(1, num_segs+1):
        arg_mask = np.argwhere(curr_seg == nseg)
        Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
        Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
        Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
        Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

        # Turn the points into an array
        seg_pts[nseg] = np.array([Px_min, Py_max, Px_max, Py_min])
        seg_pts[nseg] = seg_pts[nseg].reshape(-1, 1, 2)
        init_seg_pts = seg_pts

    # Visualize
    if VISUALIZE:
        viz_img = curr_frame.copy()
        for nseg in range(1, num_segs+1):
            n_seg_pts = seg_pts[nseg]
            viz_img = cv.polylines(viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
            for i in range(4):
                viz_img = cv.circle(viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=seg_color[nseg-1].tolist(), thickness=-1)
        cv.imshow("curr", viz_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    # Get the features in the segmentation zone 
    curr_kps_descs = {}
    for nseg in range(1, num_segs+1):
        mask = np.zeros_like(curr_seg)
        cv.fillPoly(mask, [np.flip(seg_pts[nseg], axis=2)], 255)
        curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)
        curr_kps_descs[nseg] = (curr_kps, curr_descs)
        if VISUALIZE:
            viz_img = cv.polylines(viz_img, [np.flip(seg_pts[nseg], axis=2)], True, (255,255,255), 1)
            viz_img = cv.drawKeypoints(viz_img, curr_kps, None, color=(0,255,0))
            cv.imshow("curr", viz_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

    while frame_idx < len(input_frames)-2: 
        # Increment the frame idx
        frame_idx += 1

        start = time.time()

        # If we have reached SEG_PERIOD
        if frame_idx % SEG_PERIOD == 0: 

            # Update the segmentation and current frame to frame_idx
            curr_seg = seg_frames[frame_idx]
            curr_frame = input_frames[frame_idx]
            frame_idx += 1
            
            # Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
            seg_pts = {}
            for nseg in range(1, num_segs+1):
                arg_mask = np.argwhere(curr_seg == nseg)
                Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
                Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
                Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
                Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

                # Turn the points into an array
                seg_pts[nseg] = np.array([Px_min, Py_max, Px_max, Py_min])
                seg_pts[nseg] = seg_pts[nseg].reshape(-1, 1, 2)
                init_seg_pts = seg_pts

            # Visualize
            if VISUALIZE:
                viz_img = curr_frame.copy()
                for nseg in range(1, num_segs+1):
                    n_seg_pts = seg_pts[nseg]
                    viz_img = cv.polylines(viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
                    for i in range(4):
                        viz_img = cv.circle(viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=seg_color[nseg-1].tolist(), thickness=-1)
                cv.imshow("curr", viz_img)
                cv.waitKey(0)
                cv.destroyAllWindows()

            # Get the features in the segmentation zone 
            curr_kps_descs = {}
            for nseg in range(1, num_segs+1):
                mask = np.zeros_like(curr_seg)
                cv.fillPoly(mask, [np.flip(seg_pts[nseg], axis=2)], 255)
                curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)
                curr_kps_descs[nseg] = (curr_kps, curr_descs)
                if VISUALIZE:
                    viz_img = curr_frame.copy()
                    viz_img = cv.polylines(viz_img, [np.flip(seg_pts[nseg], axis=2)], True, (255,255,255), 1)
                    viz_img = cv.drawKeypoints(viz_img, curr_kps, None, color=(0,255,0))
                    cv.imshow("curr", viz_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                
        
        # Get the next frame 
        prev_frame = curr_frame.copy()
        prev_kps_descs = curr_kps_descs.copy()
        curr_frame = input_frames[frame_idx]

        if VISUALIZE:
            mask = np.zeros_like(curr_seg)
            for nseg in range(1, num_segs+1):
                cv.fillPoly(mask, [np.flip(seg_pts[nseg], axis=2)], nseg*(255//num_segs))
            gif_mask_frames.append(Image.fromarray(mask))
            cv.imshow("Mask", mask)
            cv.waitKey(0)
            cv.destroyAllWindows()

        # Get the orb features 
        curr_kps, curr_descs = orb.detectAndCompute(curr_frame, None)
        ious_gt = []
        ious = []
        final_img = curr_frame.copy()
        for nseg in range(1, num_segs+1):
            (prev_kps, prev_descs) = prev_kps_descs[nseg]
            if VISUALIZE:
                viz_img = curr_frame.copy()
                viz_img = cv.polylines(viz_img, [np.flip(seg_pts[nseg], axis=2)], True, (255,255,255), 1)
                viz_img = cv.drawKeypoints(viz_img, prev_kps, None, color=(0,255,0))
                cv.imshow("prev", viz_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
            matches = matcher.match(prev_kps_descs[nseg][1], curr_descs)
            viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
            matches = sorted(matches, key=lambda x: x.distance)

            n_init_seg_pts = init_seg_pts[nseg]
            n_seg_pts = seg_pts[nseg]

            if VISUALIZE:
                final_img = cv.drawMatches(prev_frame, prev_kps_descs[nseg][0], curr_frame, curr_kps, viz_matches, None) 
                final_img = cv.polylines(final_img, [np.flip(n_init_seg_pts, axis=2)], True, (255,255,255), 1)
                for i in range(4):
                    final_img = cv.circle(final_img, (n_init_seg_pts[i,0,1], n_init_seg_pts[i,0,0]), 5, color=seg_color[nseg-1].tolist(), thickness=-1)
                    final_img = cv.circle(final_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=seg_color[(nseg-1)+num_segs].tolist(), thickness=-1)

            # Find the set of points in the segmentation zone with good correspondences
            u_pts = []
            v_pts = []
            seg_curr_kps = ()
            seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)

            n_seg_pts = seg_pts[nseg]
            for match in matches:
                if curr_kps[match.trainIdx].pt[0] < n_seg_pts[3,0,1] or curr_kps[match.trainIdx].pt[0] > n_seg_pts[1,0,1] or curr_kps[match.trainIdx].pt[1] < n_seg_pts[0,0,0] or curr_kps[match.trainIdx].pt[1] > n_seg_pts[2,0,0]:
                    continue 
                else:
                    u_pts.append(prev_kps[match.queryIdx].pt)
                    v_pts.append(curr_kps[match.trainIdx].pt)
                    seg_curr_kps = seg_curr_kps + (curr_kps[match.trainIdx],)
                    seg_curr_descs = np.vstack((seg_curr_descs, curr_descs[match.trainIdx]))
                # ret = cv.pointPolygonTest(n_seg_pts, (curr_kps[match.trainIdx].pt[1], curr_kps[match.trainIdx].pt[0]), False)
                # if ret < 0 and DEBUG:
                #     viz_img = curr_frame.copy()
                #     viz_img = cv.circle(viz_img, (int(curr_kps[match.trainIdx].pt[0]), int(curr_kps[match.trainIdx].pt[1])), 5, color=(0,0,255), thickness=-1)
                #     viz_img = cv.polylines(viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
                #     cv.imshow("curr", viz_img)
                #     cv.waitKey(0)
                #     cv.destroyAllWindows()
                #     continue

                # elif ret >= 0 and DEBUG:
                #     print("good match")
                #     print(curr_kps[match.trainIdx].pt)
                #     viz_img = curr_frame.copy()
                #     viz_img = cv.circle(viz_img, (int(curr_kps[match.trainIdx].pt[0]), int(curr_kps[match.trainIdx].pt[1])), 5, color=(0,255,0), thickness=-1)
                #     viz_img = cv.polylines(viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
                #     cv.imshow("curr", viz_img)
                #     cv.waitKey(0)
                #     cv.destroyAllWindows()  

                # if ret >= 0:
                #     u_pts.append(prev_kps[match.queryIdx].pt)
                #     v_pts.append(curr_kps[match.trainIdx].pt)
                #     seg_curr_kps = seg_curr_kps + (curr_kps[match.trainIdx],)
                #     seg_curr_descs = np.vstack((seg_curr_descs, curr_descs[match.trainIdx]))

            u_pts = np.flip(np.float32(u_pts).reshape(-1, 1, 2), axis=2)
            v_pts = np.flip(np.float32(v_pts).reshape(-1, 1, 2), axis=2)
        
            # Get the homography defined by this set of points
            try: 
                matrix, _ = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)
                n_seg_pts = cv.perspectiveTransform(n_seg_pts.astype(np.float32), matrix)
            
            except: 
                print("Homography failed")
                continue

            # Perform the transformation on the segmentation zone to get the new segmentation zone
            prev_seg_pts = seg_pts
            n_prev_seg_pts = prev_seg_pts[nseg]

            curr_kps_descs[nseg] = (seg_curr_kps, seg_curr_descs)

            # Measure delta between the previous and current seg pts 
            delta = n_seg_pts - n_prev_seg_pts

            # Get IOU weight
            prev_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(n_prev_seg_pts.astype(np.int64), axis=2)], 255)
            curr_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(n_seg_pts.astype(np.int64), axis=2)], 255)
            iou = np.sum(np.logical_and(prev_mask, curr_mask)) / np.sum(np.logical_or(prev_mask, curr_mask))
            ious.append(iou)
            if iou < 0.5:
                print("no_update")
                n_seg_pts = n_prev_seg_pts
                continue
            n_seg_pts = n_prev_seg_pts + iou*delta
            n_seg_pts = n_seg_pts.astype(np.int64)
            seg_pts[nseg] = n_seg_pts

            # Get IOU with GT segmentation
            gt_mask = np.where(seg_frames[frame_idx]==nseg, 255, 0)
            iou_gt = np.sum(np.logical_and(gt_mask, curr_mask)) / np.sum(np.logical_or(gt_mask, curr_mask))
            ious_gt.append(iou_gt)
        
        iou = np.array(ious).mean()
        iou_gt = np.array(ious_gt).mean()
        print("Average IOU: ", iou, ious)
        print("Average GT IOU: ", iou_gt, ious_gt)

        IOU_est.append(iou)
        IOU_gt.append(iou_gt)

        end = time.time()
        loop_time.append((end-start)*1000)
        print("Time: ", (end-start)*1000, "ms")

        if VISUALIZE:
            for nseg in range(1, num_segs):
                shifted_seg_pts = seg_pts[nseg].copy()
                shifted_seg_pts[:,0,1] += curr_frame.shape[1]
                final_img = cv.polylines(final_img, [np.flip(shifted_seg_pts, axis=1)], True, (255,255,255), 1)
                n_seg_pts = seg_pts[nseg]
                for i in range(4):
                    final_img = cv.circle(final_img, (curr_frame.shape[1] + n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=seg_color[nseg-1].tolist(), thickness=-1)
            gif_frames.append(Image.fromarray(final_img))

            # Show the final image 
            cv.imshow("Matches", final_img)
            cv.waitKey(0)
            cv.destroyAllWindows()
    return IOU_est, IOU_gt, loop_time

# fig, ax = plt.subplots(1,2)
# ax[0].plot(IOU_est)
# ax[0].plot(IOU_gt)
# ax[1].plot(loop_time)
# ax[0].legend(["Estimated IOU", "GT IOU", "loop time"])
# ax[0].set_xlabel("Frame")
# ax[0].set_ylabel("IOU")
# ax[1].set_xlabel("Frame")
# ax[1].set_ylabel("Time (ms)")
# ax[0].set_title("IOU vs Frame")
# ax[1].set_title("Loop Time vs Frame")
# plt.show()


# if len(gif_frames) != 0:
#     frame_one = gif_frames[0]
#     frame_one.save(f'output/sampled_output_{test}/{test}_matching.gif', format="GIF", append_images=gif_frames,
#                    save_all=True, duration=100, loop=0)
#     frame_one = gif_mask_frames[0]
#     frame_one.save(f'output/sampled_output_{test}/{test}_mask.gif', format="GIF", append_images=gif_mask_frames,
#                    save_all=True, duration=100, loop=0)
#     frame_one = viz_combined_frames[0]
#     frame_one.save(f'output/sampled_output_{test}/{test}_gt.gif', format="GIF", append_images=viz_combined_frames,
#                    save_all=True, duration=100, loop=0)
# print("Done")


if __name__ == "__main__":

    num_processes = 4
    PROCESS_DIRS = get_process_dirs(num_processes)
    breakpoint()

    with Pool(num_processes) as p:
        ious, ious_gt, loop_time = p.map(partial(propogate_segmentations, input_frames=INPUT_FRAMES), PROCESS_DIRS)
