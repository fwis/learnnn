## Introduction
We are trying to build a ResNet-based network for rebuilding I, AoP and DoLP from input DoFP polarization images.
This work is still in an exploration stage.
The code is based on Pytorch.

## Notes
1. You can use 'generate_labels' to process your dataset.
2. The 'test' is for generating full resolution images from a real DoFP image.

## TODO
1. Employ CAM into our network.
2. Create polarized remote sensing dataset.
3. Apply our network on remote sensing.

# Data
The DoLP and S0 was normalized in range (0,1), and the AoP was normalized in range (0,pi/2).

## Citation
The ForkNet model is cited from:
[Xianglong Zeng, Yuan Luo, Xiaojing Zhao, and Wenbin Ye, "An end-to-end fully-convolutional neural network for division of focal plane sensors to reconstruct S0, DoLP, and AoP," Opt. Express 27, 8566-8577 (2019)](https://www.osapublishing.org/oe/abstract.cfm?uri=oe-27-6-8566)
