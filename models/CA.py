import torch
import torch.nn as nn


from models.common import Conv, Bottleneck


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, inp, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out

class C3CASM(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=True, g=1,
                 e=0.5, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion #iscyy

        super(C3CASM, self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.CA = CoordAtt(2 * c_)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.channel_adjust = nn.Conv2d(2 * c_, c2, kernel_size=1, stride=1, padding=0)
        # self.m = nn.Sequential(*[CB2d(c_) for _ in range(n)])
        self.sam = SpatialAttention(kernel_size)
        self.m = nn.Sequential(*[Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)])

    def forward(self, x):
        out = torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1)

        out = self.CA(out)# C3 concat之后加入CA
        out = self.channel_adjust(out)

        # out = self.sam(out)
        spatial_attention_map = self.sam(out)

        out = out * spatial_attention_map.expand_as(out)

        out = self.cv3(out)
        return out

# class CASM(nn.Module):
#     def __init(self, in_channels, out_channels):
#         super(CASM, self).__init()
#         self.ca = C3CASM()  # CA 注意力机制
#         self.sam = SpatialAttention()  # SAM 注意力机制
#
#     def forward(self, x):
#         ca_out = self.ca(x) * x
#         # 然后经过SAM
#         sam_out = self.sam(ca_out) * ca_out
#         # 进行逐元素相乘
#         # out = ca_out * sam_out
#
#         return sam_out



if __name__ == '__main__':
    input = torch.randn(512, 512, 7, 7)
    pna = C3CASM(512, 512)
    output = pna(input)
    print(output.shape)
