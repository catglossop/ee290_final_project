import os 
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
        self.seg_sub = rospy.Subscriber("/segmentation/image_raw", Image, self.seg_callback)
        self.input_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.input_callback)
        self.reset_sub = rospy.Subscriber("/camera/reset", Empty, self.reset_callback)
        self.viz_img = Image()
        self.annotate_pub = rospy.Publisher("/image_annotated", Image, queue_size=10)
        self.mask_img = Image()
        self.mask_pub = rospy.Publisher("/mask", Image, queue_size=10)
        self.cv_bridge = CvBridge()
        self.seg_updated = False
        
        # Feature tracking 
        self.orb = cv.ORB_create()
        self.matcher = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

        # Tracking
        self.iou_est = []
        self.iou_gt = []
        self.loop_time = []
        self.eval_count = 0

        self.curr_frame = None
        self.prev_frame = None
        self.curr_seg = None

    def seg_callback(self, msg):
        self.prev_seg = self.curr_seg
        self.curr_seg = self.cv_bridge.imgmsg_to_cv2(msg, encoding="passthrough")
        self.num_segs = np.max(np.unique(self.curr_seg))
        self.seg_updated = True
        self.seg_color = np.random.randint(0, 255, (self.num_segs*2, 3))
    
    def reset_callback(self, msg):

        print("IOU: ", np.mean(self.iou_est))
        print("GT IOU: ", np.mean(self.iou_gt))
        print("Loop time: ", np.mean(self.loop_time))
        print("Saving plot to output/perf_eval_{}.png".format(self.eval_count))

        fig, ax = plt.subplots(1,2)
        ax[0].plot(self.IOU_est)
        ax[0].plot(self.IOU_gt)
        ax[1].plot(self.loop_time)
        ax[0].legend(["Estimated IOU", "GT IOU", "loop time"])
        ax[0].set_xlabel("Frame")
        ax[0].set_ylabel("IOU")
        ax[1].set_xlabel("Frame")
        ax[1].set_ylabel("Time (ms)")
        ax[0].set_title("IOU vs Frame")
        ax[1].set_title("Loop Time vs Frame")
        plt.savefig(f'output/perf_eval_{self.eval_count}.png')

        self.IOU_est = []
        self.IOU_gt = []
        self.loop_time = []

    def input_callback(self, msg):

        self.start = time.time()

        self.seg_pts = {}
        if self.seg_updated:
            for nseg in range(1, self.num_segs+1):
                arg_mask = np.argwhere(self.curr_seg == nseg)
                Px_max = arg_mask[np.argmax(arg_mask[:,0]), :]
                Px_min = arg_mask[np.argmin(arg_mask[:,0]), :]
                Py_max = arg_mask[np.argmax(arg_mask[:,1]), :]
                Py_min = arg_mask[np.argmin(arg_mask[:,1]), :]

                # Turn the points into an array
                self.seg_pts[nseg] = np.array([Px_min, Py_max, Px_max, Py_min])
                self.seg_pts[nseg] = self.seg_pts[nseg].reshape(-1, 1, 2)
                
            self.init_seg_pts = self.seg_pts
            self.seg_updated = False

            # Get the features in the segmentation zone 
            self.curr_kps_descs = {}
            for nseg in range(1, num_segs+1):
                mask = np.zeros_like(curr_seg)
                cv.fillPoly(mask, [np.flip(seg_pts[nseg], axis=2)], 255)
                curr_kps, curr_descs = orb.detectAndCompute(curr_frame, mask)
                self.curr_kps_descs[nseg] = (curr_kps, curr_descs)

        elif not self.seg_updated and self.curr_seg != None and self.curr_frame != None: 
            self.prev_frame = self.curr_frame
            self.curr_frame = self.cv_bridge.imgmsg_to_cv2(msg, encoding="passthrough")
            self.viz_img = self.curr_frame.copy()

            self.prev_kps_descs = self.curr_kps_descs

            # Get the orb features 
            curr_kps, curr_descs = self.orb.detectAndCompute(self.curr_frame, None)

            ious_gt, ious = [], []
            for nseg in range(1, self.num_segs+1):
                (prev_kps, prev_descs) = self.prev_kps_descs[nseg]
                matches = matcher.match(self.prev_kps_descs[nseg][1], curr_descs)
                viz_matches = sorted(matches, key=lambda x: x.distance)[:10]
                matches = sorted(matches, key=lambda x: x.distance)

                n_init_seg_pts = self.init_seg_pts[nseg]
                n_seg_pts = self.seg_pts[nseg]

                if self.VISUALIZE:
                    self.viz_img = cv.drawMatches(self.prev_frame, self.prev_kps_descs[nseg][0], self.curr_frame, curr_kps, viz_matches, None) 
                    self.viz_img = cv.polylines(self.final_img, [np.flip(n_init_seg_pts, axis=2)], True, (255,255,255), 1)
                    for i in range(4):
                        self.viz_img = cv.circle(self.viz_img, (n_init_seg_pts[i,0,1], n_init_seg_pts[i,0,  0]), 5, color=self.seg_color[nseg-1].tolist(), thickness=-1)
                        self.viz_img = cv.circle(self.viz_img, (n_seg_pts[i,0,1], n_seg_pts[i,0,0]), 5, color=self.seg_color[(nseg-1)+self.num_segs].tolist(), thickness=-1)

                # Find the set of points in the segmentation zone with good correspondences
                u_pts = []
                v_pts = []
                seg_curr_kps = ()
                seg_curr_descs = np.array([], dtype=np.uint8).reshape(0, 32)
                for match in matches:
                    if curr_kps[match.trainIdx].pt[0] < n_seg_pts[3,0,1] or curr_kps[match.trainIdx].pt[0] > n_seg_pts[1,0,1] or curr_kps[match.trainIdx].pt[1] < n_seg_pts[0,0,0] or curr_kps[match.trainIdx].pt[1] > n_seg_pts[2,0,0]:
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
                    n_seg_pts = cv.perspectiveTransform(n_seg_pts.astype(np.float32), matrix)
                
                except: 
                    print("Homography failed")
                    return

                # Perform the transformation on the segmentation zone to get the new segmentation zone
                self.prev_seg_pts = self.seg_pts
                n_prev_seg_pts = self.prev_seg_pts[nseg]

                self.curr_kps_descs[nseg] = (seg_curr_kps, seg_curr_descs)

                # Measure delta between the previous and current seg pts 
                delta = n_seg_pts - n_prev_seg_pts

                # Get IOU weight
                prev_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(n_prev_seg_pts.astype(np.int64), axis=2)], 255)
                self.curr_mask = cv.fillPoly(np.zeros_like(curr_seg), [np.flip(n_seg_pts.astype(np.int64), axis=2)], 255)
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
            
            self.end = time.time()
            self.loop_time.append((self.end-self.start)*1000)
            self.IOU_est.append(np.array(ious).mean())
            self.IOU_gt.append(np.array(ious_gt).mean())

        else: 
            print("Waiting for first segmentation frame...")
            return

        self.mask_img = self.cv_bridge.cv2_to_imgmsg(self.curr_mask, encoding="passthrough")
        self.mask_pub.publish(self.mask_img)

        self.annotate_img = self.cv_bridge.cv2_to_imgmsg(self.viz_img, encoding="passthrough")
        self.annotate_pub.publish(self.annotate_img)

def main():

    rospy.init_node('multi_obj_tracking_node', anonymous=True)
    multi_obj_tracking_node = MultiObjectTrackingNode()

    rospy.spin()


if __name__ == '__main__':
    main()
