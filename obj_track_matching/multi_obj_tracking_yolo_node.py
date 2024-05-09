import os, glob
import cv2 as cv
import numpy as np
import imageio as iio
import matplotlib.pyplot as plt
import copy
import time
# from PIL import Image 
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge # INSTALL ON PI
from std_msgs.msg import Float32, Empty



class MultiObjectTrackingNode:

    def __init__(self):

        self.VISUALIZE = True
        self.DEBUG = False
        self.test = "drift"
        self.seg_sub = rospy.Subscriber("/yolo/segmentation/image_raw", Image, self.seg_callback)
        self.gt_sub = rospy.Subscriber("/yolo/ground_truth/image_raw", Image, self.gt_callback)
        self.input_sub = rospy.Subscriber("/yolo/camera/color/image_raw", Image, self.input_callback)
        self.reset_sub = rospy.Subscriber("/yolo/camera/reset", Empty, self.reset_callback)
        self.viz_img = Image()
        self.annotate_pub = rospy.Publisher("/yolo/image_annotated", Image, queue_size=10)
        self.mask_img = Image()
        self.mask_pub = rospy.Publisher("/yolo/mask", Image, queue_size=10)
        self.cv_bridge = CvBridge()
        self.seg_updated = False
        
        # Feature tracking 
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Tracking
        self.IOU_est = []
        self.IOU_gt = []
        self.loop_time = []
        self.loop_clock = []
        self.eval_count = 0
        self.MODE = "ASYNC"

        self.curr_frame = None
        self.prev_frame = None
        self.curr_seg = None
        self.gt_seg = None
        self.initialized = False

        # self.ious_per_fps = []
        # self.ious_gt_per_fps = []
        # self.loop_time_per_fps = []
        # self.loop_clock_per_fps = []
        self.centroid_diff = []

        for f in glob.glob("output/perf_eval_*.jpg"):
            os.remove(f)

    def resize_polygon(self, poly):
        poly_new = poly.copy()
        centroid = np.mean(poly_new, axis=0)

        # scale the points
        poly_new = ((poly_new - centroid) * self.scale + centroid).astype(np.int64)
        # visualize
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

    def seg_callback(self, msg):
        self.prev_seg = self.curr_seg
        self.curr_seg = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.num_segs = np.max(np.unique(self.curr_seg))
        self.seg_updated = True
        self.seg_color = np.random.randint(0, 255, (self.num_segs*2, 3))
    
    def gt_callback(self, msg):
        self.gt_seg = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.gt_num_segs = np.max(np.unique(self.gt_seg))
    
    def reset_callback(self, msg):
        print("IN RESET CALLBACK")
        # print("IOU: ", np.mean(self.IOU_est))
        # print("GT IOU: ", np.mean(self.IOU_gt))
        # print("Loop time: ", np.mean(self.loop_time))
        if self.VISUALIZE:
            fig, ax = plt.subplots(1,2, figsize=(20,10))
            ax[0].plot(self.IOU_est)
            ax[0].plot(self.IOU_gt)
            ax[1].plot(self.loop_time)
            ax[0].legend(["Estimated IOU", "GT IOU"])
            ax[0].set_xlabel("Frame")
            ax[0].set_ylabel("IOU")
            ax[1].set_xlabel("Frame")
            ax[1].set_ylabel("Time (ms)")
            ax[0].set_title("IOU vs Frame")
            ax[1].set_title("Loop Time vs Frame")
            plt.savefig(f'output/{self.test}_perf_eval_{self.eval_count}.png')

        # self.IOU_gt = [iou for iou in self.IOU_gt if not np.isnan(iou)]
        # self.ious.append(np.mean(self.IOU_est))
        # self.ious_gt_per_fps.append(np.mean(self.IOU_gt))
        # self.loop_time_per_fps.append(np.mean(self.loop_time))
        # self.loop_clock_per_fps.append(np.mean(self.loop_clock))
        # self.IOU_est = []
        # self.IOU_gt = []
        # self.loop_time = []
        # self.loop_clock = []
        
        data_arr = np.array([self.IOU_est, self.IOU_gt, self.loop_time, self.loop_clock, self.centroid_diff])
        np.save(f"output/{self.test}_perf_eval.npy", data_arr)
        print("DONE EVAL")

    def input_callback(self, msg):
        self.prev_frame = self.curr_frame
        self.curr_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
        self.viz_img = self.curr_frame.copy()
        self.start = time.time()
        self.start_tick = cv.getTickCount()

        
        if self.seg_updated:
            self.seg_pts = {}
            for nseg in range(1, self.num_segs+1):
                curr_mask = np.where(self.curr_seg==nseg, 255, 0)
                self.seg_pts[nseg] = self.get_polygon(curr_mask)
                self.prev_seg_pts = self.seg_pts

                if self.MODE == "ASYNC":
                    # Compute segmentation region centroid 
                    curr_centroid = np.mean(self.prev_seg_pts[nseg], axis=0)
                    seg_centroid = np.mean(self.seg_pts[nseg], axis=0)
                    img = self.curr_frame.copy()
                    img = cv.polylines(img, [np.flip(self.prev_seg_pts[nseg], axis=2)], True, (255,255,0), 1)
                    img = cv.polylines(img, [np.flip(self.seg_pts[nseg], axis=2)], True, (0,255,255), 1)
                    self.seg_pts[nseg] = self.seg_pts[nseg] + (curr_centroid - seg_centroid).astype(np.int64)

                self.init_seg_pts = self.seg_pts
            
            if self.DEBUG:
                for nseg in range(1, self.num_segs+1):
                    n_seg_pts = self.seg_pts[nseg]
                    self.viz_img = cv.polylines(self.viz_img, [np.flip(n_seg_pts, axis=2)], True, (255,255,255), 1)
                    for i in range(4):
                        self.viz_img = cv.circle(self.viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
                
                self.viz_msg = self.cv_bridge.cv2_to_imgmsg(self.viz_img, encoding="passthrough")
                self.annotate_pub.publish(self.viz_msg)
                
            self.init_seg_pts = self.seg_pts
            self.seg_updated = False

            # Get the features in the segmentation zone 
            self.curr_kps_descs = {}
            for nseg in range(1, self.num_segs+1):
                mask = np.zeros_like(self.curr_seg)
                cv.fillPoly(mask, [np.flip(self.seg_pts[nseg], axis=2)], 255)
                curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, mask)
                self.curr_kps_descs[nseg] = (curr_kps, curr_descs)
            
            if not self.initialized:
                self.initialized = True

        elif not self.seg_updated and self.curr_seg is not None and self.curr_frame is not None and self.initialized: 
            self.prev_frame = self.curr_frame
            self.curr_frame = np.frombuffer(msg.data, dtype=np.uint8).reshape(msg.height, msg.width, -1)
            self.viz_img = self.curr_frame.copy()
            self.curr_mask_out = np.zeros_like(self.curr_seg)

            self.prev_kps_descs = self.curr_kps_descs

            # Get the orb features 
            curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, None)

            ious_gt, ious, centroid_diff = [], [], []
            for nseg in range(1, self.num_segs+1):

                n_init_seg_pts = self.init_seg_pts[nseg]
                n_seg_pts = self.seg_pts[nseg]

                resized_mask = np.zeros_like(self.curr_seg)
                resized_seg_pts = self.resize_polygon(n_seg_pts)
                resized_mask = cv.fillPoly(resized_mask, [np.flip(resized_seg_pts, axis=2)], 255)

                curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, resized_mask) 
                (prev_kps, prev_descs) = self.prev_kps_descs[nseg]

                matches = self.matcher.match(self.prev_kps_descs[nseg][1], curr_descs)
                # viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
                matches = sorted(matches, key=lambda x: x.distance)

                if self.VISUALIZE:
                    self.viz_img = cv.drawMatches(self.prev_frame, self.prev_kps_descs[nseg][0], self.curr_frame, curr_kps, viz_matches, None) 
                    self.viz_img = cv.polylines(self.viz_img, [np.flip(n_init_seg_pts, axis=2)], True, (255,255,255), 1)
                    for i in range(4):
                        self.viz_img = cv.circle(self.viz_img, (n_init_seg_pts[i,0,1], n_init_seg_pts[i,0,  0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
                        self.viz_img = cv.circle(self.viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[(nseg-1)+self.num_segs].tolist(), thickness=-1)
                
                    self.viz_msg = self.cv_bridge.cv2_to_imgmsg(self.viz_img, encoding="passthrough")
                    self.annotate_pub.publish(self.viz_msg)

                # Find the set of points in the segmentation zone with good correspondences
                # u_pts = []
                # v_pts = []
                # seg_curr_kps = ()
                # seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)
                # for match in matches:
                #     if curr_kps[match.trainIdx].pt[0] < n_seg_pts[3,0,1] or curr_kps[match.trainIdx].pt[0] > n_seg_pts[1,0,1] or curr_kps[match.trainIdx].pt[1] < n_seg_pts[0,0,0] or curr_kps[match.trainIdx].pt[1] > n_seg_pts[2,0,0]:
                #         continue 
                #     else:
                #         u_pts.append(prev_kps[match.queryIdx].pt)
                #         v_pts.append(curr_kps[match.trainIdx].pt)
                #         seg_curr_kps = seg_curr_kps + (curr_kps[match.trainIdx],)
                #         seg_curr_descs = np.vstack((seg_curr_descs, curr_descs[match.trainIdx]))

                # u_pts = np.flip(np.float32(u_pts).reshape(-1, 1, 2), axis=2)
                # v_pts = np.flip(np.float32(v_pts).reshape(-1, 1, 2), axis=2)

                u_pts = np.flip(np.float32([prev_kps[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2), axis=2)
                v_pts = np.flip(np.float32([curr_kps[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2), axis=2)
            
                # Get the homography defined by this set of points
                try: 
                    matrix, _ = cv.findHomography(u_pts, v_pts, cv.RANSAC, 5.0)
                    n_seg_pts = cv.perspectiveTransform(n_seg_pts.astype(np.float32), matrix)
                except: 
                    return

                # Perform the transformation on the segmentation zone to get the new segmentation zone
                self.prev_seg_pts = self.seg_pts
                n_prev_seg_pts = self.prev_seg_pts[nseg]

                self.curr_kps_descs[nseg] = (seg_curr_kps, seg_curr_descs)

                # Measure delta between the previous and current seg pts 
                delta = n_seg_pts - n_prev_seg_pts
                delta_vel = np.mean(delta, axis=0)
                self.scale = np.abs(delta_vel/self.seg_pts[nseg].mean(axis=0)).mean() + 1.0

                # Get IOU weight
                self.prev_mask = cv.fillPoly(np.zeros_like(self.curr_seg), [np.flip(n_prev_seg_pts.astype(np.int64), axis=2)], 255)
                self.curr_mask = cv.fillPoly(np.zeros_like(self.curr_seg), [np.flip(n_seg_pts.astype(np.int64), axis=2)], 255)
                iou = np.sum(np.logical_and(self.prev_mask, self.curr_mask)) / np.sum(np.logical_or(self.prev_mask, self.curr_mask))
                ious.append(iou)
                if iou < 0.5:
                    n_seg_pts = n_prev_seg_pts
                    continue
                n_seg_pts = n_prev_seg_pts + iou*delta
                n_seg_pts = n_seg_pts.astype(np.int64)
                self.seg_pts[nseg] = n_seg_pts

                # Get IOU with GT segmentation
                self.gt_mask = np.where(self.gt_seg==nseg, 255, 0)
                iou_gt = np.sum(np.logical_and(self.gt_mask, self.curr_mask)) / np.sum(np.logical_or(self.gt_mask, self.curr_mask))
                ious_gt.append(iou_gt)

                # Get centroid difference with GT segmentation
                gt_mask = np.where(self.seg_frames[self.frame_idx]==nseg, 255, 0)
                gt_seg_pts = self.get_polygon(gt_mask)
                gt_centroid = np.mean(gt_seg_pts, axis=0)
                curr_centroid = np.mean(n_seg_pts, axis=0)
                centroid_diff.append(np.linalg.norm(gt_centroid - curr_centroid))  

                self.curr_mask_out = cv.fillPoly(self.curr_mask_out, [np.flip(n_seg_pts, axis=2)], int(nseg*(255//self.num_segs)))
            
            self.end = time.time()
            self.end_tick = cv.getTickCount()
            self.loop_time.append((self.end-self.start)*1000)
            self.loop_clock.append((self.end_tick-self.start_tick)/cv.getTickFrequency()*1000)
            self.IOU_est.append(np.array(ious).mean())
            self.IOU_gt.append(np.array(ious_gt).mean())
            self.centroid_diff.append(np.array(centroid_diff).mean())

            self.mask_img = self.cv_bridge.cv2_to_imgmsg(self.curr_mask_out, encoding="passthrough")
            self.mask_pub.publish(self.mask_img)

            self.annotate_img = self.cv_bridge.cv2_to_imgmsg(self.viz_img, encoding="passthrough")
            self.annotate_pub.publish(self.annotate_img)

        else: 
            print("Waiting for first segmentation frame...")
            return


def main():

    rospy.init_node('yolo_multi_obj_tracking_node', anonymous=True)
    multi_obj_tracking_node = MultiObjectTrackingNode()

    rospy.spin()


if __name__ == '__main__':
    main()