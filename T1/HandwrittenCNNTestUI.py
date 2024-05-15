import sys
import torch
import torch.nn as nn
from PyQt6.QtWidgets import QApplication, QMainWindow, QHBoxLayout, QVBoxLayout, QWidget, QPushButton, QTextEdit, QLayout
from PyQt6.QtGui import QPainter, QPen, QColor, QPalette, QPaintEvent,QPixmap
from PyQt6.QtCore import Qt, QPoint
from PIL import Image as PilImage
from torchvision import transforms

# 定义CNN模型
class CNNNet(nn.Module):
    def __init__(self):
        super(CNNNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5, stride=1, padding=2)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, stride=1, padding=2)
        self.fc1 = nn.Linear(7*7*64, 1024)
        self.fc2 = nn.Linear(1024, 10)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 7*7*64)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# 加载模型
model = CNNNet()
model.load_state_dict(torch.load('T1/mnist_cnn.pth'))
model.eval()

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
def predict_digit(image):
    image = load_and_preprocess_image(image)
    with torch.no_grad():
        output = model(image)
        predicted_digit = output.argmax().item()
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

        # 创建Clear按钮
        self.clearButton = QPushButton('Clear', self)
        self.clearButton.setFixedSize(200, 40)
        self.clearButton.clicked.connect(self.handwrittenWidget.clear)

        # 创建Recognize按钮
        self.recognizeButton = QPushButton('Recognize', self)
        self.recognizeButton.setFixedSize(200, 40)
        self.recognizeButton.clicked.connect(self.onRecognizeClicked)

        # 创建 Result 文本框
        self.result_textbox = QTextEdit(self)
        self.result_textbox.setFixedSize(200, 40)

        # 设置布局
        layout = QVBoxLayout()
        
        topLayout = QHBoxLayout()
        topLayout.addStretch(1)
        topLayout.addWidget(self.handwrittenWidget)
        topLayout.addStretch(1)

        bottomLayout = QHBoxLayout()
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

    def onRecognizeClicked(self):
        # 使用grab()方法获取子控件的内容
        qpixmap = self.handwrittenWidget.getPixmap()
        #qpixmap.save('d.bmp');
        # 将QPixmap转换为QImage
        qimage = qpixmap.toImage()

        # 将QImage转换为PIL的Image
        pil_image = PilImage.fromqimage(qimage)

        # 调整图像大小为28x28
        pil_image = pil_image.resize((28, 28), PilImage.Resampling.LANCZOS)
        #pil_image.show()
        # 使用predict_digit函数进行预测
        predicted_digit = predict_digit(pil_image)
        print(f'The predicted digit is: {predicted_digit}')
        self.result_textbox.setText(str(predicted_digit))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    mainWindow  = HandwrittenMainWindow()
    mainWindow.show()
    sys.exit(app.exec())
