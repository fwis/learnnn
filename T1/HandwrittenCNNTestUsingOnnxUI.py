import sys
import os
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog
from PyQt6.QtGui import QPainter, QPen, QColor, QPalette, QPaintEvent,QPixmap
from PyQt6.QtCore import Qt, QPoint
from PIL import Image as PilImage
from torchvision import transforms
import numpy as np
import onnxruntime

onnx_file_path = 'T1/mnist_cnn.onnx'
onnx_session = onnxruntime.InferenceSession(onnx_file_path)

# 定义一个函数，用于读取图片并将其转换为模型可以接受的格式
def load_and_preprocess_image(image):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=1),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    image = transform(image).unsqueeze(0)  # 增加一个维度，因为模型期望的输入是[batch_size, channels, height, width]
    return image

# 定义一个函数，用于预测图片中的数字
def predict_digit(pil_image):
    input_name = onnx_session.get_inputs()[0].name
    output_name = onnx_session.get_outputs()[0].name
    tensor_image = load_and_preprocess_image(pil_image)
    np_image = tensor_image.numpy()
    result = onnx_session.run([output_name], {input_name: np_image})[0]
    predicted_digit = np.argmax(result, axis=1).item()
    #with torch.no_grad():
    #    output = model(image)
    #    predicted_digit = output.argmax().item()
    return predicted_digit

class HandwrittenWidget(QWidget):
    def __init__(self, parent=None):
        super(HandwrittenWidget, self).__init__(parent)
        self.setFixedSize(280, 280)  # 设置固定大小
        palette = self.palette()  # 获取当前窗口的调色板

        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)  # 设置Window角色的颜色为黑色

        self.setPalette(palette)  # 应用新的调色板到窗口
        self.setAutoFillBackground(True)
        
        self.drawing = False
        self.points_group = []
        self.points = []
        self.points_group.append(self.points)

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.white, 13, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if self.drawing:
            for points in self.points_group:
                p1 = None
                for p2 in points:
                    if p1 != None:
                        painter.drawLine(p1, p2)
                    p1 = p2
        else:
            #print('nnnn')
            pass

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.points = []
            self.points_group.append(self.points)
            self.drawing = True
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            #self.drawing = False
            pass

    def mouseMoveEvent(self, event):
        if (event.buttons() & Qt.MouseButton.LeftButton) and self.drawing:
            self.points.append(event.pos())
            self.update()

    def clear(self):
        self.points_group.clear()
        self.points.clear()
        self.update()

    def getPixmap(self):
        # 创建一个QPixmap，其大小和Widget相同
        pixmap = QPixmap(self.size())
        # 用QPainter将Widget的内容绘制到pixmap上
        painter = QPainter(pixmap)
        painter.drawPixmap(QPoint(0, 0), pixmap)
        self.render(painter)
        return pixmap
    
class HandwrittenMainWindow(QMainWindow):
    def __init__(self):
        super(HandwrittenMainWindow, self).__init__()
        self.setWindowTitle('手写数字识别程序')
        self.setGeometry(100, 100, 800, 600)

        # 创建手绘组件实例
        self.handwrittenWidget = HandwrittenWidget()

        # 创建LoadModel按钮
        self.loadButton = QPushButton('Load Model', self)
        self.loadButton.setFixedSize(200, 40)
        self.loadButton.clicked.connect(self.onLoadModelClicked)

        # 创建Clear按钮
        self.clearButton = QPushButton('Clear', self)
        self.clearButton.setFixedSize(150, 40)
        self.clearButton.clicked.connect(self.handwrittenWidget.clear)

        # 创建Recognize按钮
        self.recognizeButton = QPushButton('Recognize', self)
        self.recognizeButton.setFixedSize(150, 40)
        self.recognizeButton.clicked.connect(self.onRecognizeClicked)

        # 创建 Result 文本框
        self.result_textbox = QTextEdit(self)
        self.result_textbox.setFixedSize(100, 40)

        # 设置布局
        layout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addStretch(1)
        topLayout.addWidget(self.handwrittenWidget)
        topLayout.addStretch(1)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.loadButton)
        bottomLayout.addWidget(self.clearButton)
        bottomLayout.addWidget(self.recognizeButton)
        bottomLayout.addWidget(self.result_textbox)
        #  bottomLayout.setSizeConstraint(QLayout.SetFixedSize)

        layout.addLayout(topLayout)
        layout.addLayout(bottomLayout)

        # 设置中心窗口
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)
    
    def onLoadModelClicked(self):
        # 设置初始目录
        current_path = os.getcwd()
        file_dialog = QFileDialog()
        file_dialog.setDirectory(f"{current_path}/T1/")
        print(f"{current_path}/T1/")
        # 设置文件过滤器，仅显示 .data 扩展名的文件
        file_dialog.setNameFilter("Data Files (*.onnx)")
        #file_dialog.selectNameFilter("Data Files (*.onnx)")  # 可选，确保默认选择这个过滤器

        # 弹出对话框并等待用户选择
        if file_dialog.exec():
            # 获取选择的文件路径
            selected_files = file_dialog.selectedFiles()
            onnx_file_path = selected_files[0]
            onnx_session = onnxruntime.InferenceSession(onnx_file_path)
            self.setTitle('手写体数字识别, 当前使用模型: ' + onnx_file_path)
        else:
            print("No file selected.")

    def onRecognizeClicked(self):
        # 使用grab()方法获取子控件的内容
        qpixmap = self.handwrittenWidget.getPixmap()
        #qpixmap.save('d.bmp');
        # 将QPixmap转换为QImage
        qimage = qpixmap.toImage()

        # 将QImage转换为PIL的Image
        pil_image = PilImage.fromqimage(qimage)

        np_image = np.array(pil_image)

        # 找到白色区域的索引
        white_pixels = np.where(np_image > 200)

        # 计算白色区域的最小外包矩形
        min_x, max_x = np.min(white_pixels[1]), np.max(white_pixels[1])
        min_y, max_y = np.min(white_pixels[0]), np.max(white_pixels[0])

        # 计算扩展后的正方形尺寸
        size = max(max_x - min_x + 1, max_y - min_y + 1)
        size = int(size * 1.2)

        # 计算粘贴位置，使原始内容位于正方形中央
        paste_pos = ((size - (max_x - min_x + 1)) // 2, (size - (max_y - min_y + 1)) // 2)

        sub_np_img = np_image[min_y:max_y,min_x:max_x]
        # 将原始白色区域裁剪并粘贴到正方形中心
        sub_pil_img = PilImage.fromarray(sub_np_img)
        #sub_pil_img.show()

        # 创建一个新的正方形图像，初始填充为黑色
        result_pil_img = PilImage.new('L', (size, size), 0)
        result_pil_img.paste(sub_pil_img, paste_pos)
        #result_pil_img.show()

        # 使用predict_digit函数进行预测
        predicted_digit = predict_digit(result_pil_img)
        print(f'The predicted digit is: {predicted_digit}')
        self.result_textbox.setText(str(predicted_digit))

    def setTitle(self, title):
        self.setWindowTitle(title)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow  = HandwrittenMainWindow()
    onnx_session.get_modelmeta()
    mainWindow.setTitle(onnx_file_path)
    mainWindow.show()
    sys.exit(app.exec())
