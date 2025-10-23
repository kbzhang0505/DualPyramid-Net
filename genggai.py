import os

# 设置标签文件的目录路径
label_dir = r"C:\Users\admin\Desktop\hao_yolo\yolov5-master\DD\labels\val"

# 需要修改的错误部分
old_str = "firc_close_"
new_str = "firc_cloth_"

# 获取所有 .txt 标签文件
label_files = [f for f in os.listdir(label_dir) if f.endswith('.txt')]

# 记录是否找到匹配的文件
found = False

# 批量重命名
for old_name in label_files:
    if old_str in old_name:  # 仅修改包含 `firec_close_` 的文件
        found = True
        new_name = old_name.replace(old_str, new_str)  # 替换错误的部分
        old_path = os.path.join(label_dir, old_name)
        new_path = os.path.join(label_dir, new_name)

        os.rename(old_path, new_path)
        print(f"重命名: {old_name} -> {new_name}")

if not found:
    print("未找到匹配的文件，请检查文件名是否正确！")

print("批量修改完成！")

