import matplotlib.pyplot as plt
import numpy as np 

# segs = ["parachute", "bird_of_paradise", "frog", "drift", "penguin"]
# segs = ["frog", "drift", "parachute", "bird_of_paradise"]
segs = ["penguin", "frog", "drift", "parachute", "bird_of_paradise"]
mode = "sync"
colors = ["b", "g", "r", "c", "m"]

fig, ax = plt.subplots(1,3, figsize=(15,5))

for idx, seg in enumerate(segs): 
    print(seg)
    path = f"/home/ee290/ee290_ws/src/ee290_final_project/obj_track_matching/{seg}_output/{seg}_perf_eval_{mode}.npy"
    print(path)

    data = np.load(path, allow_pickle=True)
    truncate = 4
    periods = data[0][:truncate]
    ious_per_fps = data[1][:truncate]
    ious_gt_per_fps = data[2][:truncate]
    loop_time_per_fps = data[3][:truncate]
    loop_clock_per_fps = data[4][:truncate]
    centroid_diff_per_fps = data[5][:truncate]

    # lens = np.array([len(ious_per_fps), len(ious_gt_per_fps), len(loop_time_per_fps), len(centroid_diff_per_fps)])

    # min_len = np.min()

    ax[0].plot(periods, ious_per_fps, label=seg, color=colors[idx])
    ax[0].plot(periods, ious_gt_per_fps, '--', color=colors[idx])
    ax[1].plot(periods, loop_time_per_fps, label=seg, color=colors[idx])
    ax[2].plot(periods, centroid_diff_per_fps, label=seg, color=colors[idx])

ax[1].plot(periods, np.repeat(375, truncate), '--', label="avg. yolov8n")

ax[0].set_xlabel("Segmentation Period")
ax[1].set_xlabel("Segmentation Period")
ax[1].set_xlabel("Segmentation Period")
ax[0].set_ylabel("IOU")
ax[1].set_ylabel("Latency (ms)")
ax[2].set_ylabel("Error (pixels)")
ax[0].legend(loc="upper right")
ax[1].legend(loc="upper right")
ax[2].legend(loc="upper right")
ax[0].set_title("IOU vs Segmentation Period")
ax[1].set_title("Loop Time vs Segmentation Period")
ax[2].set_title("Avg. Centroid Difference vs Segmentation Period")
plt.savefig(f'output/perf_eval_over_periods_{mode}.png')