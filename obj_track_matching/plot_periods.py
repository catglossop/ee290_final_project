import matplotlib.pyplot as plt
import numpy as np 

# segs = ["parachute", "bird_of_paradise", "frog", "drift", "penguin"]
segs = ["drift"]

fig, ax = plt.subplots(1,2, figsize=(20,10))

for seg in segs: 
    path = f"/home/proj206a/ee290_ws/src/ee290_final_project/obj_track_matching/output/{seg}_perf_eval.npy"

    data = np.load(path, allow_pickle=True)

    periods = data[0]
    ious_per_fps = data[1]
    ious_gt_per_fps = data[2]
    loop_time_per_fps = data[3]
    loop_clock_per_fps = data[4]

    ax[0].plot(periods[:len(ious_per_fps)], ious_per_fps)
    ax[0].plot(periods[:len(ious_per_fps)], ious_gt_per_fps, '--')
    ax[1].plot(periods[:len(ious_per_fps)], loop_time_per_fps)
    ax[1].plot(periods[:len(ious_per_fps)], loop_clock_per_fps, '--')
ax[0].set_xlabel("Segmentation Period")
ax[1].set_xlabel("Segmentation Period")
ax[0].set_ylabel("IOU")
ax[1].set_ylabel("Time (ms)")
ax[0].legend(segs)
ax[1].legend(segs)
ax[0].set_title("IOU vs Segmentation Period")
ax[1].set_title("Loop Time vs Segmentation Period")
plt.savefig(f'output/perf_eval_over_periods.png')