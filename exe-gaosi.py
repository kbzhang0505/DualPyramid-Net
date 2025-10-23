import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt


class Deblur(nn.Module):
    def __init__(self, kernel_size=5, sigma=1.5):
        super().__init__()
        self.kernel_size = kernel_size
        self.sigma = sigma

    def forward(self, image):
        # 检查输入图像是否为Tensor
        if not isinstance(image, torch.Tensor):
            raise TypeError("Input image must be a torch.Tensor object")

        # 检查输入图像的维度是否正确
        if image.dim() != 4:
            raise ValueError("Input image must be a 4D tensor (batch_size, channels, height, width)")

        # 使用高斯模糊对图像进行模糊处理
        blurred_image = nn.functional.gaussian_blur(image, kernel_size=self.kernel_size, sigma=self.sigma)

        return blurred_image


# 创建一个Deblur模块的实例
deblur_module = Deblur()

# 读取输入图像
input_image_path = "input_image.jpg"
input_image = Image.open(input_image_path)

# 将输入图像转换为Tensor，并添加批量维度
transform = transforms.Compose([transforms.ToTensor()])
input_image_tensor = transform(input_image).unsqueeze(0)

# 对输入图像进行模糊处理
output_image_tensor = deblur_module(input_image_tensor)

# 将处理后的图像转换回PIL图像
output_image = transforms.ToPILImage()(output_image_tensor.squeeze(0))

# 在弹窗中显示处理后的图像
plt.imshow(output_image)
plt.axis('off')  # 关闭坐标轴
plt.show()
