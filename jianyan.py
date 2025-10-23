import os
from PIL import Image

# 定义数据集图片所在的目录
image_dirs = [
    'C:\\Users\\admin\\Desktop\\hao_yolo\\yolov5-master\\zijian\\images\\train',
    'C:\\Users\\admin\\Desktop\\hao_yolo\\yolov5-master\\zijian\\images\\val',
    'C:\\Users\\admin\\Desktop\\hao_yolo\\yolov5-master\\zijian\\images\\test'
]

# 遍历每个目录
for image_dir in image_dirs:
    if not os.path.exists(image_dir):
        print(f"目录 {image_dir} 不存在。")
        continue
    # 获取目录下的所有文件
    files = os.listdir(image_dir)
    for file in files:
        file_path = os.path.join(image_dir, file)
        # 检查文件是否为图片文件
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            try:
                # 尝试打开图片
                img = Image.open(file_path)
                img.verify()  # 验证图片的完整性
                print(f"{file_path} 是有效的图片。")
            except FileNotFoundError:
                print(f"错误：未找到图片 {file_path}。")
            except (IOError, SyntaxError) as e:
                print(f"错误：图片 {file_path} 损坏或无效，错误信息：{e}。")