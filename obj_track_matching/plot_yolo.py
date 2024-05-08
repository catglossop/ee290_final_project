import matplotlib.pyplot as plt
import numpy as np 

models = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]
params = [3.4, 11.8, 27.3, 46.0, 71.8]
latency = [375.80, 402.84, 475.05, 572.87, 674.85]
colors = ["b", "g", "r", "c", "m"]

fig, ax = plt.subplots()

for idx in range(len(params)):

    ax.plot(params[idx], latency[idx], 'o' )
    ax.text(params[idx], latency[idx], models[idx])

ax.set_title("Yolov8 Model Size vs. Latency on Jetson Xavier AGX")
ax.set_xlabel("Num. parameters (M)")
ax.set_ylabel("Latency (ms)")

plt.savefig(f'output/yolo_perf.png')