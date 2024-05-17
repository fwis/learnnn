# 经典的卷积神经网络实现的手写体识别

## 基于2024/05 比较新的环境：
- Python 3.12
- PyQt6
- Pytorch2.3
- onnx-1.16.0

## 卷积神经网络
- Q1: 神经网络中的卷积和图像卷积有什么不同？

    [视频链接](https://www.youtube.com/watch?v=XdTn5md3qTM)

    [视频链接](https://www.bilibili.com/video/BV16N411y7cV/)

    个人理解：
    1. 多通道处理方式不同：NetConv并不是对Source的每个Channel单独Conv后合并。理解一下 神经网络中 1x1 卷积
    2. 结构不同：神经网络中的卷积是大于二维的，而图像卷积是二维的
