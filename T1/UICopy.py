import sys
import os
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QTextEdit, QFileDialog, QLabel, QMessageBox, QCheckBox
from PyQt6.QtGui import QPainter, QPen, QColor, QPalette, QPaintEvent, QPixmap, QImage
from PyQt6.QtCore import Qt, QPoint
from PIL import Image as PilImage
from torchvision import transforms
import numpy as np
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
import cv2
import NNs as mynns

pth_file_path = 'D:/Neural_Networks/learnnn/T2/mnist_ResNet2.pth'
num_blocks = [1, 1, 0, 0]
num_classes = 10
mothed_name = 'ResNet2'
pth_model = mynns.ResNet(mynns.ResNet.Residual, num_blocks, num_classes=num_classes)
pth_model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
pth_model.eval()  # 设置模型为评估模式

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
        self.image = None  # 存储加载的图像

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setPen(QPen(Qt.GlobalColor.white, 13, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
        if self.image:
            painter.drawPixmap(QPoint(0, 0), self.image) 
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
        self.image = None
        self.update()

    def getPixmap(self):
        # 创建一个QPixmap，其大小和Widget相同
        pixmap = QPixmap(self.size())
        # 用QPainter将Widget的内容绘制到pixmap上
        painter = QPainter(pixmap)
        painter.drawPixmap(QPoint(0, 0), pixmap)
        self.render(painter)
        return pixmap
    
    def save_image(self, file_path):
        pixmap = self.getPixmap()
        pixmap.save(file_path)
        
    def set_image(self, image):
        self.image = image
        self.update()
    
class CamWidget(QWidget):
    def __init__(self, parent=None):
        super(CamWidget, self).__init__(parent)
        self.setFixedSize(280, 280)  # 设置固定大小
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.setLayout(self.layout)
        
        palette = self.palette()  # 获取当前窗口的调色板
        palette.setColor(QPalette.ColorRole.Window, Qt.GlobalColor.black)  # 设置Window角色的颜色为黑色
        self.setPalette(palette)  # 应用新的调色板到窗口
        self.setAutoFillBackground(True)

    def set_heatmap(self, heatmap):
        black_image = np.zeros((280, 280, 3), dtype=np.uint8)  # 三通道 RGB 图像
        black_height, black_width, black_channels = black_image.shape
        black_qimage = QImage(black_image.data, black_width, black_height, 3 * black_width, QImage.Format.Format_RGB888)
        black_pixmap = QPixmap.fromImage(black_qimage)
        self.label.setPixmap(black_pixmap)
        # 将 numpy 数组转换为 QImage
        heatmap = cv2.resize(heatmap, (280, 280), interpolation=cv2.INTER_LINEAR)
        height, width, channels = heatmap.shape
        qimage = QImage(heatmap.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)
        self.label.repaint()
    
    def clear(self):
        black_image = np.zeros((280, 280, 3), dtype=np.uint8)  # 三通道 RGB 图像
        height, width, channels = black_image.shape
        qimage = QImage(black_image.data, width, height, 3 * width, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage)
        self.label.setPixmap(pixmap)

def predict_digit(pil_image):
    global mothed_name
    tensor_image = load_and_preprocess_image(pil_image) # (1, 1, 28, 28)
    with torch.no_grad():
        output = pth_model(tensor_image)
        if mothed_name == 'ResNet2':
            target_layers = [pth_model.layer2[-1].conv3]
        elif mothed_name == 'FCN':
            target_layers = [pth_model.conv5]
        elif mothed_name == 'CNN':
            target_layers = [pth_model.conv1, pth_model.conv2]
        predicted_digit = output.argmax(dim=1).item()

    heatmap = generate_cam_heatmap(target_layers, tensor_image)

    return predicted_digit, heatmap

def generate_cam_heatmap(target_layer, tensor_image):
    
    # 生成Grad-CAM heatmap（对一个图像）
    cam = GradCAM(model=pth_model, target_layers=target_layer)
    mask = cam(input_tensor=tensor_image)[0, :] # (H, W, C)
    
    # 转换成图像
    gray_img = tensor_image.squeeze().numpy()
    rgb_img_hwc = np.repeat(gray_img[:, :, np.newaxis], 3, axis=2) # (H, W, C)
    rgb_img_hwc = cv2.normalize(rgb_img_hwc, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
    visualization_c3u8_rgb = show_cam_on_image(rgb_img_hwc, mask, use_rgb=True)
    
    return visualization_c3u8_rgb

class HandwrittenMainWindow(QMainWindow):
    def __init__(self):
        super(HandwrittenMainWindow, self).__init__()
        self.setWindowTitle('手写数字识别程序')
        self.setGeometry(100, 100, 800, 600)  # 增加宽度以适应热力图窗口
        self.handwrittenWidget = HandwrittenWidget()
        self.camWidget = CamWidget()  # 创建热力图组件
        self.processed_image = None
        self.heatmap = None

        # 创建LoadModel按钮
        self.loadModelButton = QPushButton('Load Model', self)
        self.loadModelButton.setFixedSize(150, 40)
        self.loadModelButton.clicked.connect(self.onLoadModelClicked)

        # 创建LoadImage按钮
        self.loadImageButton = QPushButton('Load Image', self)
        self.loadImageButton.setFixedSize(150, 40)
        self.loadImageButton.clicked.connect(self.onLoadImageClicked)
        
        # 创建Clear按钮
        self.clearButton = QPushButton('Clear', self)
        self.clearButton.setFixedSize(150, 40)
        self.clearButton.clicked.connect(self.handwrittenWidget.clear)
        self.clearButton.clicked.connect(self.clearProcessedImage)
        self.clearButton.clicked.connect(self.camWidget.clear)

        # 创建Recognize按钮
        self.recognizeButton = QPushButton('Recognize', self)
        self.recognizeButton.setFixedSize(150, 40)
        self.recognizeButton.clicked.connect(self.onRecognizeClicked)

        # 创建 Save 按钮
        self.saveButton = QPushButton('Save Image', self)
        self.saveButton.setFixedSize(150, 40)
        self.saveButton.clicked.connect(self.onSaveImageClicked)
        
        # 创建 Save Processed Image 按钮
        self.saveProcessedButton = QPushButton('Save Processed Image', self)
        self.saveProcessedButton.setFixedSize(150, 40)
        self.saveProcessedButton.clicked.connect(self.onSaveProcessedImageClicked)
        
        # 创建 Save Heatmap 按钮
        self.saveHeatmapButton = QPushButton('Save Heatmap', self)
        self.saveHeatmapButton.setFixedSize(150, 40)
        self.saveHeatmapButton.clicked.connect(self.onSaveHeatmapClicked)
        
        # 创建 Result 文本框
        self.result_textbox = QTextEdit(self)
        self.result_textbox.setFixedSize(100, 40)

        # 设置布局
        layout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addStretch(1)
        topLayout.addWidget(self.handwrittenWidget)
        topLayout.addWidget(self.camWidget)  # 添加热力图组件
        topLayout.addStretch(1)

        bottomLayout = QHBoxLayout()
        bottomLayout.addWidget(self.loadModelButton)
        bottomLayout.addWidget(self.loadImageButton)
        bottomLayout.addWidget(self.clearButton)
        bottomLayout.addWidget(self.recognizeButton)
        bottomLayout.addWidget(self.saveButton)
        bottomLayout.addWidget(self.saveProcessedButton)
        bottomLayout.addWidget(self.saveHeatmapButton)
        bottomLayout.addWidget(self.result_textbox)
        #  bottomLayout.setSizeConstraint(QLayout.SetFixedSize)

        layout.addLayout(topLayout)
        layout.addLayout(bottomLayout)

        # 设置中心窗口
        centralWidget = QWidget()
        centralWidget.setLayout(layout)
        self.setCentralWidget(centralWidget)
        
    def clearProcessedImage(self):
        self.processed_image = None
        self.heatmap = None
    
    def onLoadModelClicked(self):
        global pth_model
        global mothed_name
        # 设置初始目录
        current_path = os.getcwd()
        file_dialog = QFileDialog()
        file_dialog.setDirectory(f"{current_path}/T1/")
        print(f"{current_path}/T1/")
        # 设置文件过滤器，仅显示 .data 扩展名的文件
        file_dialog.setNameFilter("Data Files (*.pth)")
        #file_dialog.selectNameFilter("Data Files (*.onnx)")  # 可选，确保默认选择这个过滤器

        # 弹出对话框并等待用户选择
        if file_dialog.exec():
            # 获取选择的文件路径
            selected_files = file_dialog.selectedFiles()
            pth_file_path = selected_files[0]
            if 'ResNet2' in pth_file_path:
                mothed_name = 'ResNet2'
                num_blocks = [1, 1, 0, 0]
                pth_model = mynns.ResNet(mynns.ResNet.Residual, num_blocks, num_classes)
                pth_model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
            elif 'fcn' in pth_file_path:
                mothed_name = 'FCN'
                pth_model = mynns.FCN(num_classes)
                pth_model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
            elif 'cnn' in pth_file_path:
                mothed_name = 'CNN'
                pth_model = mynns.CNNNet()
                pth_model.load_state_dict(torch.load(pth_file_path, map_location=torch.device('cpu')))
            self.setTitle('手写体数字识别, 当前使用模型: ' + pth_file_path)
        else:
            print("No file selected.")

    def onLoadImageClicked(self):
        # 打开文件对话框以选择图像文件
        file_path, _ = QFileDialog.getOpenFileName(self, "打开图像", "", "图像文件 (*.png *.jpg *.bmp)")
        if file_path:
            pil_image = PilImage.open(file_path).convert('L')  # 转换为灰度图像
            pil_image_resized = pil_image.resize((280, 280))  # 调整大小
        
            # 显示选中的图像
            self.handwrittenWidget.clear()
            self.handwrittenWidget.repaint()  # 清除之前的内容
            qimage = QImage(pil_image_resized.tobytes(), pil_image_resized.width, pil_image_resized.height, QImage.Format.Format_Grayscale8)
            pixmap = QPixmap.fromImage(qimage)
            # self.handwrittenWidget.update()
            self.handwrittenWidget.set_image(pixmap)
            
        else:
            print("未选择文件。")

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

        if len(white_pixels[0]) == 0 or len(white_pixels[1]) == 0:
            QMessageBox.warning(self, "Warning", "Please draw a figure or load an image first.")
            return
        
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
        self.processed_image = result_pil_img
        #result_pil_img.show()

        # 使用predict_digit函数进行预测
        predicted_digit, heatmap = predict_digit(result_pil_img)
        self.heatmap = heatmap
        print(f'The predicted digit is: {predicted_digit}')
        self.result_textbox.setText(str(predicted_digit))
        self.camWidget.set_heatmap(heatmap)  # 显示热力图

    def onSaveProcessedImageClicked(self):
        if self.processed_image:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Processed Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                processed_image_resized = self.processed_image.resize((28, 28))
                processed_image_resized.save(file_path)
            else:
                print("No file selected.")
        else:
            QMessageBox.warning(self, "No Processed Image", "There is no processed image to save. Please recognize an image first.")
    
    def onSaveHeatmapClicked(self):
        if self.heatmap is not None:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Heatmap", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                pil_heatmap = PilImage.fromarray(self.heatmap)
                big_heatmap = pil_heatmap.resize((280, 280))
                big_heatmap.save(file_path)
            else:
                print("No file selected.")
        else:
            QMessageBox.warning(self, "No Heatmap", "There is no Heatmap to save. Please recognize an image first.")
    
    def onSaveImageClicked(self):
        qimage = self.handwrittenWidget.getPixmap()
        pil_image = PilImage.fromqimage(qimage)
        np_image = np.array(pil_image)
        
        white_pixels = np.where(np_image > 200)
        if len(white_pixels[0]) == 0 or len(white_pixels[1]) == 0:
            QMessageBox.warning(self, "Warning", "Please draw figure first.")
        else:
            file_path, _ = QFileDialog.getSaveFileName(self, "Save Image", "", "PNG Files (*.png);;JPEG Files (*.jpg)")
            if file_path:
                self.handwrittenWidget.save_image(file_path)
    
    def setTitle(self, title):
        self.setWindowTitle(title)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow = HandwrittenMainWindow()
    mainWindow.setTitle('手写数字识别程序')
    mainWindow.show()
    sys.exit(app.exec())
