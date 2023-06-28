import shutil
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import sys
import cv2
import torch
import os.path as osp
from train import SELFMODEL
import numpy as np
from torch import nn
from torchutils import get_torch_transforms
from PIL import Image

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

class MainWindow(QTabWidget):
    def __init__(self):
        # 初始化界面
        super().__init__()
        self.setWindowTitle('垃圾分类系统')
        self.resize(1200, 800) # 设置窗口大小
        self.setWindowIcon(QIcon("images/UI/lufei.png"))
        # 图片读取进程
        self.output_size = 480 # 设置上传图片的输出大小
        self.img2predict = ""
        self.origin_shape = () # 设置图片原始大小,图片上传会变形,所以记录一下
        self.model_path = "checkpoints/resnet50d_pretrained_224/resnet50d_87epochs_accuracy0.99994_weights.pth"  # 模型路径
        self.classes_names = ['有害垃圾', '厨余垃圾', '其他垃圾', '可回收物']  #类名
        self.img_size = 224  # 输入图片大小
        self.model_name = "resnet50d"  # 模型名称
        self.num_classes = len(self.classes_names)  # 类别数目

        # 加载并初始化模型
        # 创建一个 SELFMODEL 类型的模型，并命名为 self.model_name，输出类别数为 self.num_classes，不使用预训练的参数
        model = SELFMODEL(model_name=self.model_name, out_features=self.num_classes, pretrained=False)
        weights = torch.load(self.model_path,
                             map_location=torch.device('cpu')) # 加载模型权重，并将其放在 CPU 上
        model.load_state_dict(weights) # 将加载的权重赋值给模型
        model.eval() # 将模型设置为评估模式,用于优化训练而添加的网络层会被关闭，使得评估时不会发生偏移
        model.to(device)
        self.model = model # 将加载的模型赋值给 self.model

        # 加载数据处理
        data_transforms = get_torch_transforms(img_size=self.img_size) # 加载数据处理
        self.valid_transforms = data_transforms['val'] # 获取验证集使用的数据转换器,对训练集、验证集和测试集进行数据预处理
        self.initUI() # 初始化 UI

    '''
    ***界面初始化***
    '''
    def initUI(self):
        # todo 识别分类子界面
        font_title = QFont('楷体', 16)
        font_main = QFont('楷体', 14)
        # 图片识别界面, 两个按钮，上传图片和显示结果
        img_detection_widget = QWidget() # 创建一个小部件
        img_detection_layout = QVBoxLayout() # 设置为垂直布局
        img_detection_title = QLabel("垃圾分类系统") # 标题
        img_detection_title.setFont(font_title)
        mid_img_widget = QWidget() # 创建一个小部件
        mid_img_layout = QHBoxLayout()
        # 创建了两个img标签
        self.left_img = QLabel()
        self.right_img = QLabel()

        self.left_img.setPixmap(QPixmap("images/UI/up.jpeg"))
        self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
        self.left_img.setAlignment(Qt.AlignCenter)
        self.right_img.setAlignment(Qt.AlignCenter)
        mid_img_layout.addWidget(self.left_img)
        mid_img_layout.addStretch(0)
        mid_img_widget.setLayout(mid_img_layout)
        # 创建两个按钮
        up_img_button = QPushButton("上传图片")
        det_img_button = QPushButton("分类")
        up_img_button.clicked.connect(self.upload_img) # 创建点击事件连接函数
        det_img_button.clicked.connect(self.detect_img)
        # 设置button字体
        up_img_button.setFont(font_main)
        det_img_button.setFont(font_main)
        #上传图片和开始检测按钮的样式
        self.rrr = QLabel("等待分类")
        self.rrr.setFont(font_main) # 设置字体
        up_img_button.setStyleSheet("QPushButton{color:white}"
                                    "QPushButton:hover{background-color: rgb(2,110,180);}"
                                    "QPushButton{background-color:rgb(48,124,208)}"
                                    "QPushButton{border:2px}"
                                    "QPushButton{border-radius:5px}"
                                    "QPushButton{padding:5px 5px}"
                                    "QPushButton{margin:5px 5px}")
        det_img_button.setStyleSheet("QPushButton{color:white}"
                                     "QPushButton:hover{background-color: rgb(2,110,180);}"
                                     "QPushButton{background-color:rgb(48,124,208)}"
                                     "QPushButton{border:2px}"
                                     "QPushButton{border-radius:5px}"
                                     "QPushButton{padding:5px 5px}"
                                     "QPushButton{margin:5px 5px}")
        # 将按钮、标签和小部件添加到布局中
        img_detection_layout.addWidget(img_detection_title, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(mid_img_widget, alignment=Qt.AlignCenter)
        img_detection_layout.addWidget(self.rrr)
        img_detection_layout.addWidget(up_img_button)
        img_detection_layout.addWidget(det_img_button)
        img_detection_widget.setLayout(img_detection_layout) # 布局添加到小部件中
        self.addTab(img_detection_widget, '识别分类')  # 小部件添加到窗口的选项卡


        # todo 关于系统子界面
        about_widget = QWidget()  # 创建一个小部件
        about_layout = QVBoxLayout()  # 设置为垂直布局
        about_title = QLabel('欢迎使用基于深度学习的垃圾分类器')  # 标题
        about_title.setFont(font_title)
        about_title.setAlignment(Qt.AlignCenter)
        about_img = QLabel()
        about_img.setPixmap(QPixmap('images/UI/up.jpeg'))
        about_img.setAlignment(Qt.AlignCenter)

        label_super = QLabel()  # todo 更换作者信息
        label_super.setText("<a href='https://blog.csdn.net/ECHOSON'>或者你可以在这里找到我-->肆十二</a>")
        label_super.setFont(QFont('楷体', 16))
        label_super.setOpenExternalLinks(True)
        label_super.setAlignment(Qt.AlignRight)
        about_layout.addWidget(about_title)
        about_layout.addStretch()
        about_layout.addWidget(about_img)
        about_layout.addStretch()
        about_layout.addWidget(label_super)
        about_widget.setLayout(about_layout)

        self.left_img.setAlignment(Qt.AlignCenter)
        self.addTab(about_widget, '关于系统')
        #self.setTabIcon(0, QIcon('images/UI/lufei.png'))
        #self.setTabIcon(1, QIcon('images/UI/lufei.png'))

    '''
    ***上传图片***
    '''
    def upload_img(self):
        # 选择图片进行读取
        fileName, fileType = QFileDialog.getOpenFileName(self, 'Choose file', '', '*.jpg *.png *.tif *.jpeg')
        if fileName:
            # 将选择的文件复制到临时保存路径
            suffix = fileName.split(".")[-1]
            save_path = osp.join("images/tmp", "tmp_upload." + suffix)
            shutil.copy(fileName, save_path)
            # 对选择的图片进行缩放，使其符合统一的尺寸
            im0 = cv2.imread(save_path)
            resize_scale = self.output_size / im0.shape[0]
            im0 = cv2.resize(im0, (0, 0), fx=resize_scale, fy=resize_scale)
            cv2.imwrite("images/tmp/upload_show_result.jpg", im0)
            self.img2predict = fileName
            # 保存原图的尺寸
            self.origin_shape = (im0.shape[1], im0.shape[0])
            # 将图片显示在图片控件中
            self.left_img.setPixmap(QPixmap("images/tmp/upload_show_result.jpg"))
            # 上传图片之后图片重置，
            self.right_img.setPixmap(QPixmap("images/UI/right.jpeg"))
            self.rrr.setText("等待识别")

    '''
    ***检测图片***
    '''
    def detect_img(self):
        # 读取待检测的图片文件,首先要知道图片名
        source = self.img2predict
        img = Image.open(source) # 打开图片文件
        img = self.valid_transforms(img) # 对图片进行预处理
        img = img.unsqueeze(0) # 在第一维处添加一个维度,使图像的维度符合模型的输入要求
        img = img.to(device)
        output = self.model(img) # 使用训练好的模型进行预测
        label_id = torch.argmax(output).item() # 获取最大值所在的索引
        predict_name = self.classes_names[label_id] # 获取预测的类别名称
        self.rrr.setText("当前分类结果为：{}".format(predict_name)) # 将预测的结果显示在文本控件中

    '''
    ***退出***
    '''
    def closeEvent(self, event):
        reply = QMessageBox.question(self,
                                     '退出',
                                     "是否要退出程序？",
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.close()
            event.accept()
        else:
            event.ignore()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec_())
