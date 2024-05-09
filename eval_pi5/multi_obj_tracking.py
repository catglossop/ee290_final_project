"Evaluation & profiling for object tracking algorithm on a PC. Initially, using fake image stream to simulate real-time processing."
"Outputs a plot of number of calls & time taken for each subfunction" 
"For now, strip away the IOU metrics and other things"

import os 
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
from PIL import Image 
import cProfile
import pstats 
from pstats import SortKey

### SegTrackv2
data_path = "/Users/charlesgordon/Desktop/Research/290/Data/data_sets/SegTrackv2"
gt_path = os.path.join(data_path, "GroundTruth")
input_path = os.path.join(data_path, "JPEGImages")
#test = "girl"
#test = "frog"
#test = "drift"
test = "bird_of_paradise"

if test == "human":
    input_test_path = os.path.join(input_path)
    gt_test_path = os.path.join(gt_path)
else:
    input_test_path = os.path.join(input_path, test)
    gt_test_path = os.path.join(gt_path, test)

class MultiObjTracking: 

    def __init__(self, test, gt_path, input_path, image_size=(640, 360), SEG_PERIOD=10, DEBUG=False, VISUALIZE=True, PLOT=False, scale=1):
        self.image_size = image_size
        self.test = test
        self.load_data(test, gt_path, input_path)
        self.num_segs = np.max(np.asarray(self.combined_frames))
        self.seg_color = np.random.randint(0, 255, (self.num_segs*2, 3))
        self.scale = scale
        self.alpha = 1.0

        ###############
        print(f"Tracking {self.num_segs} objects in {self.test}")

        # Initialize orb feature detector and matcher
        self.orb = cv.ORB_create() 
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Initialize the segmentation points
        self.seg_pts = {}
        self.init_seg_pts = {}
        self.curr_seg = None
        self.curr_frame = None
        self.prev_frame = None
        self.prev_kps_descs = {}
        self.curr_kps_descs = {}
        self.delta_vel = 0

        # Modes 
        self.SEG_PERIOD = SEG_PERIOD
        self.DEBUG = DEBUG
        self.VISUALIZE = VISUALIZE
        self.PLOT = PLOT
        self.MODE = "SYNC"

        """
        # Initialize the IOU values
        self.IOU_est = []
        self.IOU_gt = []
        self.centroid_diff = {}
        """

        # Initialize the gif frames
        self.gif_frames = []
        self.gif_mask_frames = []

        # Initialize the loop time
        self.loop_time = []
        
        self.frame_idx = 0

        os.makedirs(f"output/sampled_output_{self.test}", exist_ok=True)

        self.prop_loop()

    def load_data(self, dataset, gt_test_path, input_test_path): 
        combined_frames = []
        viz_combined_frames = []
        input_frames = []

        for img in sorted(os.listdir(os.path.join(input_test_path))):
            frame = cv.imread(os.path.join(input_test_path, img))
            frame = cv.resize(frame, self.image_size)
            input_frames.append(frame)

        if dataset == "human":
            for img in sorted(os.listdir(os.path.join(gt_test_path))):
                frame = cv.imread(os.path.join(gt_test_path, img))
                frame = cv.resize(frame, self.image_size)
                combined_frames.append(frame[..., 0])
                viz_combined_frames.append(frame[..., 0]*(255//np.max(frame)))
        else: 
            seg_frames = {1: []}
            for sub in sorted(os.listdir(os.path.join(gt_test_path))):
                sub_path = os.path.join(gt_test_path, sub)
                if os.path.isdir(sub_path):
                    seg_frames[int(sub)] = []
                    for img in sorted(os.listdir(sub_path)):
                        frame = iio.imread(os.path.join(sub_path,img))
                        frame = cv.resize(frame, self.image_size)
                        seg_frames[int(sub)].append(frame[..., 0])
                else:
                    frame = iio.imread(sub_path)
                    frame = cv.resize(frame, self.image_size)
                    seg_frames[1].append(frame[..., 0])

            # Combine together frames
            for idx in enumerate(seg_frames[1]):
                combined_frame = np.zeros_like(seg_frames[1][idx[0]])
                viz_combined_frame = np.zeros_like(seg_frames[1][idx[0]])
                for key in seg_frames.keys():
                    combined_frame[seg_frames[key][idx[0]] == 255] = key
                    viz_combined_frame[seg_frames[key][idx[0]] == 255] = key*(255//len(seg_frames.keys()))
                combined_frames.append(combined_frame)
                viz_combined_frames.append(Image.fromarray(viz_combined_frame))
        
        self.input_frames = input_frames
        self.seg_frames = combined_frames
        self.combined_frames = combined_frames
        self.viz_combined_frames = viz_combined_frames

        return

    def resize_polygon(self, poly):
        poly_new = poly.copy()
        centroid = np.mean(poly_new, axis=0)

        # scale the points
        poly_new = ((poly_new - centroid) * self.scale + centroid).astype(np.int64)
        # visualize, off for now 
        if self.VISUALIZE:
            img = self.curr_frame.copy()
            img = cv.polylines(img, [np.flip(poly_new, axis=2)], True, (255,255,255), 1)
            img = cv.polylines(img, [np.flip(poly.reshape(-1, 1, 2), axis=2)], True, (255,255,255), 1)
            cv.imshow("seg region", img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        return poly_new
    
    def get_polygon(self, mask):
        arg_mask = np.argwhere(mask == 255)
        Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
        Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
        Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
        Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]


        # Turn the points into an array
        seg_pts = np.array([Px_min, Py_max, Px_max, Py_min])
        seg_pts = seg_pts.reshape(-1, 1, 2)

        return seg_pts

    
    def seg_update(self):
        # Update the segmentation and current frame to frame_idx
        if self.MODE == "SYNC":
            self.curr_seg = self.seg_frames[self.frame_idx]
        if self.MODE == "ASYNC":
            self.curr_seg = self.seg_frames[self.frame_idx//self.SEG_PERIOD]

        self.curr_frame = self.input_frames[self.frame_idx]

        # Find the four points of the segmentation (Px_max, Px_min, Py_max, Py_min)
        if self.frame_idx != 0:
            self.prev_seg_pts = self.seg_pts.copy()
        self.seg_pts = {}
        for nseg in range(1, self.num_segs+1):
            curr_mask = np.where(self.curr_seg==nseg, 255, 0)
            self.seg_pts[nseg] = self.get_polygon(curr_mask)

            if self.frame_idx != 0 and self.MODE == "ASYNC":
                # Compute segmentation region centroid 
                curr_centroid = np.mean(self.prev_seg_pts[nseg], axis=0)
                seg_centroid = np.mean(self.seg_pts[nseg], axis=0)
                img = self.curr_frame.copy()
                img = cv.polylines(img, [np.flip(self.prev_seg_pts[nseg], axis=2)], True, (255,255,0), 1)
                img = cv.polylines(img, [np.flip(self.seg_pts[nseg], axis=2)], True, (0,255,255), 1)
                self.seg_pts[nseg] = self.seg_pts[nseg] + (curr_centroid - seg_centroid).astype(np.int64)
                img = cv.polylines(img, [np.flip(self.seg_pts[nseg], axis=2)], True, (255,0,255), 1)
                img = cv.circle(img, (int(curr_centroid[0,1]), int(curr_centroid[0,0])), 5, color=(255,0,0), thickness=-1)
                img = cv.circle(img, (int(seg_centroid[0,1]), int(seg_centroid[0,0])), 5, color=(0,0,255), thickness=-1)
                cv.imshow("seg region", img)
                cv.waitKey(0)
                cv.destroyAllWindows()


        self.init_seg_pts = self.seg_pts
        self.frame_idx += 1

        # Visualize
        if self.VISUALIZE:
            viz_img = self.curr_frame.copy()
            for nseg in range(1, self.num_segs+1):
                if nseg not in self.seg_pts:
                    continue
                n_seg_pts = self.seg_pts[nseg]
                viz_img = cv.polylines(viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
                for i in range(4):
                    viz_img = cv.circle(viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
            cv.imshow("curr", viz_img)
            cv.waitKey(0)
            cv.destroyAllWindows()

        # Get the features in the segmentation zone 
        self.curr_kps_descs = {}
        for nseg in range(1, self.num_segs+1):
            if nseg not in self.seg_pts:
                continue
            mask = np.zeros_like(self.curr_seg)
            cv.fillPoly(mask, [np.flip(self.seg_pts[nseg], axis=2)], 255)
            curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, mask)
            self.curr_kps_descs[nseg] = (curr_kps, curr_descs)
            if self.VISUALIZE:
                viz_img = self.curr_frame.copy()
                viz_img = cv.polylines(viz_img, [np.flip(self.seg_pts[nseg], axis=2)], True, (255,255,255), 1)
                viz_img = cv.drawKeypoints(viz_img, curr_kps, None, color=(0,255,0))
                cv.imshow("curr", viz_img)
                cv.waitKey(0)
                cv.destroyAllWindows()
   
    def mask_viz(self): 
        mask = np.zeros_like(self.curr_seg)
        for nseg in range(1, self.num_segs+1):
            if nseg not in self.seg_pts:
                continue
            cv.fillPoly(mask, [np.flip(self.seg_pts[nseg], axis=2)], int(nseg*(255//self.num_segs)))
        self.gif_mask_frames.append(Image.fromarray(mask))
        cv.imshow("Mask", mask)
        cv.waitKey(0)
        cv.destroyAllWindows()
    
    def matches_viz(self, viz_matches, nseg, curr_kps, n_init_seg_pts, n_seg_pts):

        final_img = cv.drawMatches(self.prev_frame, self.prev_kps_descs[nseg][0], self.curr_frame, curr_kps, viz_matches, None)
        final_img = cv.polylines(final_img, [np.flip(n_init_seg_pts, axis=2)], True, (255,255,255), 1)
        for i in range(4):
            final_img = cv.circle(final_img, (n_init_seg_pts[i,0,1], n_init_seg_pts[i,0,0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
            final_img = cv.circle(final_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[(nseg-1)+self.num_segs].tolist(), thickness=-1)
        
        cv.imshow("Matches", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows() 
    
    def final_viz(self):
        final_img = self.curr_frame.copy()
        for nseg in range(1, self.num_segs):
            shifted_seg_pts = self.seg_pts[nseg].copy()
            shifted_seg_pts[:,0,1] += self.curr_frame.shape[1]
            final_img = cv.polylines(final_img, [np.flip(shifted_seg_pts, axis=1)], True, (255,255,255), 1)
            n_seg_pts =self.seg_pts[nseg]
            for i in range(4):
                final_img = cv.circle(final_img, (self.curr_frame.shape[1] + n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
        self.gif_frames.append(Image.fromarray(final_img))

        # Show the final image 
        cv.imshow("Matches", final_img)
        cv.waitKey(0)
        cv.destroyAllWindows()

    
    def plot_viz(self):
        fig, ax = plt.subplots(1,3, figsize=(15,5))
        ax[0].plot(self.IOU_est)
        ax[0].plot(self.IOU_gt)
        ax[1].plot(self.loop_time)
        for nseg in range(1, self.num_segs+1):
            ax[2].plot(self.centroid_diff[nseg], label=f"Object {nseg}")
        ax[0].legend(["Estimated IOU", "GT IOU"])
        ax[0].set_xlabel("Frame")
        ax[0].set_ylabel("IOU")
        ax[0].set_title("IOU vs Frame")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Time (ms)")
        ax[1].set_title("Loop Time vs Frame")
        ax[2].legend()
        ax[2].set_xlabel("Frame")
        ax[2].set_ylabel("Centroid Difference")
        ax[2].set_title("Centroid Difference vs Frame")
        if self.VISUALIZE:
            plt.show()
        if self.PLOT:
            plt.savefig(f"output/sampled_output_{self.test}/{self.test}_plot.png")

    def prop_loop(self):
        #For now, just run for 5 frames
        #while self.frame_idx < len(self.input_frames)-2:
        frame_limit = 10 
        while self.frame_idx < frame_limit:

            start = time.time()
            # If we have reached SEG_PERIOD
            if self.frame_idx % self.SEG_PERIOD == 0: 
                print("Updating Segmentation", self.frame_idx)
                self.seg_update()
            
            # Increment the frame idx
            self.frame_idx += 1
            # Get the next frame 
            self.prev_frame = self.curr_frame.copy()
            self.prev_kps_descs = self.curr_kps_descs.copy()
            self.curr_frame = self.input_frames[self.frame_idx]

            #Visualization off for this evaluation 
            if self.VISUALIZE:
                self.mask_viz()

            # Get the orb features 
            ious_gt = []
            ious = []
            final_img = self.curr_frame.copy()
            for nseg in range(1, self.num_segs+1):
                if nseg not in self.seg_pts:
                    continue
                n_init_seg_pts = self.init_seg_pts[nseg]
                n_seg_pts = self.seg_pts[nseg]

                resized_mask = np.zeros_like(self.curr_seg)
                resized_seg_pts = self.resize_polygon(n_seg_pts)
                resized_mask = cv.fillPoly(resized_mask, [np.flip(resized_seg_pts, axis=2)], 255)

                curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, resized_mask) 
                (prev_kps, prev_descs) = self.prev_kps_descs[nseg]
                if self.VISUALIZE:
                    viz_img = self.curr_frame.copy()
                    viz_img = cv.polylines(viz_img, [np.flip(self.seg_pts[nseg], axis=2)], True, (255,255,255), 1)
                    viz_img = cv.drawKeypoints(viz_img, prev_kps, None, color=(0,255,0))
                    cv.imshow("prev", viz_img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()
                try:
                    matches = self.matcher.match(self.prev_kps_descs[nseg][1], curr_descs)
                except: 
                    breakpoint()
                # viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
                matches = sorted(matches, key=lambda x: x.distance)

                if self.VISUALIZE:
                    self.matches_viz(matches, nseg, curr_kps, n_init_seg_pts, n_seg_pts)

                u_pts = np.flip(np.float32([prev_kps[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2), axis=2)
                v_pts = np.flip(np.float32([curr_kps[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2), axis=2)

                try: 
                    matrix, _ = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)
                    n_seg_pts = cv.perspectiveTransform(n_seg_pts.astype(np.float32), matrix)
                except:
                    print("Homography failed")
                    continue

                self.prev_seg_pts = self.seg_pts
                n_prev_seg_pts = self.prev_seg_pts[nseg]

                self.curr_kps_descs[nseg] = (curr_kps, curr_descs)

                #Measurements for accuracy evaluation, will affect performance so we will turn it off for now
                
                # Measure delta between the previous and current seg pts 
                delta = n_seg_pts - n_prev_seg_pts
                delta_vel = np.mean(delta, axis=0)
                self.scale = np.abs(delta_vel/self.seg_pts[nseg].mean(axis=0)).mean() + 1.0

                # Get IOU weight
                prev_mask = cv.fillPoly(np.zeros_like(self.curr_seg), [np.flip(n_prev_seg_pts.astype(np.int64), axis=2)], 255)
                curr_mask = cv.fillPoly(np.zeros_like(self.curr_seg), [np.flip(n_seg_pts.astype(np.int64), axis=2)], 255)
                iou = np.sum(np.logical_and(prev_mask, curr_mask)) / np.sum(np.logical_or(prev_mask, curr_mask))
                ious.append(iou)

                # Get IOU with GT segmentation
                gt_mask = np.where(self.seg_frames[self.frame_idx]==nseg, 255, 0)
                iou_gt = np.sum(np.logical_and(gt_mask, curr_mask)) / np.sum(np.logical_or(gt_mask, curr_mask))
                ious_gt.append(iou_gt)
                

                if self.VISUALIZE: 
                    img = self.curr_frame.copy()
                    img = cv.polylines(img, [np.flip(n_prev_seg_pts.astype(np.int64), axis=2)], True, (255,255,0), 1)
                    img = cv.polylines(img, [np.flip(n_seg_pts.astype(np.int64), axis=2)], True, (0,255,255), 1)
                    cv.imshow("seg region", img)
                    cv.waitKey(0)
                    cv.destroyAllWindows()

                if iou < 0.5:
                    print("no_update")
                    n_seg_pts = n_prev_seg_pts
                    continue

                n_seg_pts = n_prev_seg_pts + self.alpha*iou*delta
                n_seg_pts = n_seg_pts.astype(np.int64)
                self.seg_pts[nseg] = n_seg_pts

                """
                # Get IOU with GT segmentation
                gt_mask = np.where(self.seg_frames[self.frame_idx]==nseg, 255, 0)
                iou_gt = np.sum(np.logical_and(gt_mask, curr_mask)) / np.sum(np.logical_or(gt_mask, curr_mask))
                ious_gt.append(iou_gt)

                # Get centroid difference with GT segmentation
                gt_mask = np.where(self.seg_frames[self.frame_idx]==nseg, 255, 0)
                gt_seg_pts = self.get_polygon(gt_mask)
                gt_centroid = np.mean(gt_seg_pts, axis=0)
                curr_centroid = np.mean(n_seg_pts, axis=0)

                if self.centroid_diff.get(nseg) is None:
                    self.centroid_diff[nseg] = []
                    self.centroid_diff[nseg].append(np.linalg.norm(gt_centroid - curr_centroid))
                else:
                    self.centroid_diff[nseg].append(np.linalg.norm(gt_centroid - curr_centroid))
            
            iou = np.array(ious).mean()
            iou_gt = np.array(ious_gt).mean()

            if np.isnan(iou) or np.isnan(iou_gt): 
                breakpoint()
            print("Average IOU: ", iou, ious)
            print("Average GT IOU: ", iou_gt, ious_gt)

            self.IOU_est.append(iou)
            self.IOU_gt.append(iou_gt)

            end = time.time()
            self.loop_time.append((end-start)*1000)
            print("Time: ", (end-start)*1000, "ms")

            if self.VISUALIZE:
                self.final_viz()

        if self.PLOT or self.VISUALIZE:
            self.plot_viz()
        
        if len(self.gif_frames) != 0:
            frame_one = self.gif_frames[0]
            frame_one.save(f'output/sampled_output_{self.test}/{self.test}_matching.gif', format="GIF", append_images=self.gif_frames,
                        save_all=True, duration=100, loop=0)
            frame_one = self.gif_mask_frames[0]
            frame_one.save(f'output/sampled_output_{self.test}/{self.test}_mask.gif', format="GIF", append_images=self.gif_mask_frames,
                        save_all=True, duration=100, loop=0)
            frame_one = self.viz_combined_frames[0]
            frame_one.save(f'output/sampled_output_{self.test}/{self.test}_gt.gif', format="GIF", append_images=self.viz_combined_frames,
                        save_all=True, duration=100, loop=0)  
        print("Done")  
        """

def main():
    with cProfile.Profile() as pr: 
        print("Running Multi Object Tracking...")
        frame_limit = 10
        MOT = MultiObjTracking(test, gt_test_path, input_test_path, VISUALIZE=False)
    results = pstats.Stats(pr)
    output_file_name = "profiles_bird_10frames.prof" 
    results.dump_stats(output_file_name)
    results.sort_stats(SortKey.CUMULATIVE).print_stats()



if __name__ == "__main__":
    main()
