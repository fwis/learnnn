## 介绍
一个全卷积神经网络，从DoFP输入直接生成I, AoP, DoLP
网络结构仿照ForkNet，原来的工作是在tensorflow上进行的，这个网络是基于pytorch的。

## Citation
![ForkNet](https://github.com/AGroupofProbiotocs/ForkNet/blob/master/ForkNet.jpg)  
[Xianglong Zeng, Yuan Luo, Xiaojing Zhao, and Wenbin Ye, "An end-to-end fully-convolutional neural network for division of focal plane sensors to reconstruct S0, DoLP, and AoP," Opt. Express 27, 8566-8577 (2019)](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-6-8566)

## Notes
1. 四个方向的偏振强度图像，可以通过运行“utils”文件中的“generate_labels”文件来生成输入和标签。