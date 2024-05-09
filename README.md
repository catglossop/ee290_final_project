# ee290_final_project

Contributors: Catherine Glossop, Charles Gordon, Bear Häon 
##Abstract: 

In robotics, we have conflicting needs in our hardware and software specifications. On one end, we require hardware and software that can operate at a high frequency, delivering control commands that are reactive to changing observations and dynamic environments. However, we require that this hardware is energy efficient, compact, and, often, inexpensive. Therefore, we must optimize our utilization of this hardware to maximize throughput. In this project, we aim to tackle a problem of this nature. We develop a method for rapidly propagating segmentation masks between high-accuracy segmentation mask updates, allowing for fast robot control that balances accuracy and efficiency can perform agnostic of the underlying segmentation method.  

##Our Method: 
Our method for mask propagation focuses on tracking polygonal regions that estimate segmented regions from high-precision models. By propagating these estimations using feature tracking, we can update our segmentation masks between frames until the next segmentation is available, at which point we can re-compute the polygonal estimate of the segmentation masks. 

Video demonstration of our visual servoing [here] [https://drive.google.com/file/d/1hGNR7iuMU5kFicrv-t5E3NKvd-CRywWv/view?usp=sharing] 


