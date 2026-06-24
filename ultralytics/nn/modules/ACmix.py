import torch
import torch.nn as nn


# ---------------------- 辅助函数（不变）----------------------
def position(H, W, is_cuda=True, dtype=torch.float32):
    loc_w = torch.linspace(-1.0, 1.0, W, dtype=dtype)
    loc_h = torch.linspace(-1.0, 1.0, H, dtype=dtype)

    loc_w = loc_w.unsqueeze(0).repeat(H, 1)  # [H, W]
    loc_h = loc_h.unsqueeze(1).repeat(1, W)  # [H, W]

    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)  # [1, 2, H, W]
    if is_cuda and torch.cuda.is_available():
        loc = loc.cuda(non_blocking=True)
    return loc


def stride(x, stride):
    _b, _c, _h, _w = x.shape
    return x[:, :, ::stride, ::stride]


def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)


def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.0)


# ---------------------- 修正后的ACmix类 ----------------------
class ACmix(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super().__init__()

        self.in_planes = in_planes
        self.out_planes = out_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation

        self.rate1 = nn.Parameter(torch.Tensor(1))
        self.rate2 = nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head
        # 断言确保out_planes能被head整除
        assert self.out_planes % self.head == 0, f"out_planes({self.out_planes}) must be divisible by head({self.head})"

        # 生成Q/K/V的卷积层
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=1)
        # 位置编码处理层
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        # 注意力计算组件
        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = nn.Softmax(dim=1)

        # 生成深度卷积核参数的全连接层
        self.fc = nn.Conv2d(3 * self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)

        # 深度卷积层（分组卷积，关键参数正确）
        self.dep_conv = nn.Conv2d(
            in_channels=self.kernel_conv * self.kernel_conv * self.head_dim,  # 1152 = 3*3*128
            out_channels=self.out_planes,  # 256
            kernel_size=self.kernel_conv,  # 3
            bias=True,
            groups=self.head_dim,  # 128（分组数=头维度）
            padding=1,
            stride=stride,
        )

        self.reset_parameters()

    def reset_parameters(self):
        # 初始化平衡权重
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)

        # ---------------------- 核心修正：权重维度生成 ----------------------
        # 1. 初始权重形状：[1, kernel_conv², 3, 3]（1=单个输出通道，9=每组输入通道数）
        kernel = torch.zeros(
            1,  # 第一维度：输出通道数的基数（后续repeat到256）
            self.kernel_conv**2,  # 第二维度：每组输入通道数（9=3×3）
            self.kernel_conv,  # 卷积核高度
            self.kernel_conv,  # 卷积核宽度
        )
        # 2. 赋值为单位矩阵对应的卷积核（确保初始为类恒等映射）
        for i in range(self.kernel_conv**2):
            # 索引修正：[输出通道基数, 每组输入通道索引, 核高, 核宽]
            kernel[0, i, i // self.kernel_conv, i % self.kernel_conv] = 1.0

        # 3. 扩展到所有输出通道：[1,9,3,3] → [256,9,3,3]（正确形状）
        kernel = kernel.repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)

        # 4. 初始化偏置
        if self.dep_conv.bias is not None:
            init_rate_0(self.dep_conv.bias)
        # print(f"权重实际形状：{self.dep_conv.weight.shape}（需为[256,9,3,3]）")

    def forward(self, x):
        input_dtype = x.dtype
        b, _c, h, w = x.shape
        h_out, w_out = h // self.stride, w // self.stride
        scaling = float(self.head_dim) ** -0.5

        # 1. 生成Q/K/V
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)

        # 2. 处理位置编码
        pos_enc = position(h, w, x.is_cuda, dtype=input_dtype)
        pos_enc = pos_enc.to(input_dtype, non_blocking=True)
        pe = self.conv_p(pos_enc)  # [1, 128, h, w]

        # 3. 注意力路径（多头注意力）
        q_att = q.view(b * self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b * self.head, self.head_dim, h, w)
        v_att = v.view(b * self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        # 展开K和位置编码
        unfold_k = self.unfold(self.pad_att(k_att)).view(b * self.head, self.head_dim, self.kernel_att**2, h_out, w_out)
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att**2, h_out, w_out)

        # 计算注意力权重并加权V
        att = (q_att.unsqueeze(2) * (unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1)
        att = self.softmax(att)
        out_att = self.unfold(self.pad_att(v_att)).view(b * self.head, self.head_dim, self.kernel_att**2, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        # 4. 卷积路径（深度卷积）
        # 融合Q/K/V生成卷积核参数：[b, 3×head, head_dim, h×w] → [b, head_dim, 3×head, h×w] → [b, 1152, h, w]
        f_all = self.fc(
            torch.cat(
                [
                    q.view(b, self.head, self.head_dim, h * w),
                    k.view(b, self.head, self.head_dim, h * w),
                    v.view(b, self.head, self.head_dim, h * w),
                ],
                1,
            )
        )
        # 重塑为深度卷积的输入形状：[b, 1152, h, w]（1152=3×3×128）
        f_conv = f_all.permute(0, 2, 1, 3).reshape(b, self.kernel_conv**2 * self.head_dim, h, w)
        f_conv = f_conv.to(input_dtype)

        # 深度卷积前验证输入通道数
        # print(f"深度卷积输入形状：{f_conv.shape}（需为[b, 1152, h, w]）")
        out_conv = self.dep_conv(f_conv)

        # 5. 融合输出
        return (self.rate1 * out_att + self.rate2 * out_conv).to(input_dtype)
