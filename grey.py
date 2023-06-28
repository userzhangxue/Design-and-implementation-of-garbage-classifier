import os
from PIL import Image

# 指定原始彩色图片文件夹路径
image_folder = r'E:\a\dataset'
# 指定灰度图像的存储文件夹路径
gray_folder = r'E:\a\dataset_grey'

# 检查并创建灰度图像存储文件夹
if not os.path.exists(gray_folder):
    os.mkdir(gray_folder)

# 循环遍历原始彩色图片文件夹中的所有图片
for filename in os.listdir(image_folder):
    # 检查文件是否为图像文件
    if filename.endswith('.jpg') or filename.endswith('.png') or filename.endswith('.jpeg'):
        # 打开原始彩色图像
        img = Image.open(os.path.join(image_folder, filename))
        # 转换图像为灰度图像
        gray_img = img.convert('L')
        # 生成灰度图像文件保存路径
        gray_path = os.path.join(gray_folder, filename)
        # 将灰度图像保存到灰度图片存储文件夹中
        gray_img.save(gray_path)
