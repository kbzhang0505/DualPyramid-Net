import os

# 定义 images 和 labels 文件夹的路径
images_base_dir = 'C:\\Users\\admin\\Desktop\\hao_yolo\\yolov5-master\\zijian\\images'
labels_base_dir = 'C:\\Users\\admin\\Desktop\\hao_yolo\\yolov5-master\\zijian\\labels'

# 定义图片文件的扩展名
image_extensions = ('.jpg', '.jpeg', '.png')

# 定义数据集的子集，如 train、val、test
subsets = ['train', 'val', 'test']

# 遍历每个子集
for subset in subsets:
    labels_dir = os.path.join(labels_base_dir, subset)
    images_dir = os.path.join(images_base_dir, subset)

    # 检查 labels 文件夹是否存在
    if not os.path.exists(labels_dir):
        print(f"Labels 文件夹 {labels_dir} 不存在，跳过该子集。")
        continue

    # 检查 images 文件夹是否存在
    if not os.path.exists(images_dir):
        print(f"Images 文件夹 {images_dir} 不存在，跳过该子集。")
        continue

    # 获取 labels 文件夹下的所有标注文件
    label_files = os.listdir(labels_dir)

    for label_file in label_files:
        # 提取标注文件对应的图片文件名
        image_name = os.path.splitext(label_file)[0]

        # 尝试找到对应的图片文件
        image_found = False
        for ext in image_extensions:
            image_path = os.path.join(images_dir, image_name + ext)
            if os.path.exists(image_path):
                image_found = True
                break

        if not image_found:
            print(f"未找到与标注文件 {os.path.join(labels_dir, label_file)} 对应的图片。")
        else:
            print(f"标注文件 {os.path.join(labels_dir, label_file)} 对应的图片已找到。")
