from model_resnet import resnet50, resnet34, resnet18
from timm.models.layers import DropPath,  trunc_normal_
from einops import  repeat
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

class Conv(nn.Module):
    def __init__(self, inp_dim, out_dim, kernel_size=3, stride=1, bn=False, relu=True, bias=True, group=1):
        super(Conv, self).__init__()
        self.inp_dim = inp_dim
        self.conv = nn.Conv2d(inp_dim, out_dim, kernel_size, stride, padding=(kernel_size-1)//2, bias=bias)
        self.relu = None
        self.bn = None
        if relu:
            self.relu = nn.ReLU(inplace=True)
        if bn:
            self.bn = nn.BatchNorm2d(out_dim)

    def forward(self, x):
        assert x.size()[1] == self.inp_dim, "{} {}".format(x.size()[1], self.inp_dim)
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            # [batch_size, channels, height, width]
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

def drop_path_f(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.

    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path_f(x, self.drop_prob, self.training)

class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class UMF_GTSS(nn.Module):
    def __init__(self, in_channel1, in_channel2, out_channel):
        super(UMF_GTSS, self).__init__()

        self.act_fn = nn.ReLU()

        self.layer_rec1 = nn.Sequential(
            nn.Conv2d(in_channel1, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            self.act_fn
        )
        self.layer_rec2 = nn.Sequential(
            nn.Conv2d(in_channel2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            self.act_fn
        )

        self.layer_uncer_1 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1)
        )
        self.layer_uncer_2 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel // 2, kernel_size=3, stride=1, padding=1)
        )

        self.layer_cat = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channel),
            self.act_fn
        )

    def forward(self, x1, x2):
        x_rec1 = self.layer_rec1(x1)
        x_rec2 = self.layer_rec2(x2)

        x_in1 = x_rec1.unsqueeze(1)
        x_in2 = x_rec2.unsqueeze(1)
        x_cat = torch.cat((x_in1, x_in2), dim=1)

        x_max = x_cat.max(dim=1)[0]

        # 计算不确定度（基于博弈理论的方法）
        uncert_out1 = self.layer_uncer_1(x_max)
        uncert_out1 = F.softmax(uncert_out1, dim=1)

        # 博弈理论中的策略选择与对抗均衡
        # 将softmax输出视为博弈中的策略，计算博弈的"收益"（这里假设对抗博弈中的平衡策略）
        strategy_1 = uncert_out1
        payoff_1 = strategy_1 * (1 - strategy_1)  # 假设博弈收益为策略的对抗性差异

        x_max_att = 1 - torch.sum(payoff_1, dim=1)  # 使用对抗博弈的平衡差异来量化不确定度

        x_mul = x_rec1 * x_rec2

        uncert_out2 = self.layer_uncer_2(x_mul)
        uncert_out2 = F.softmax(uncert_out2, dim=1)

        # 计算博弈策略2
        strategy_2 = uncert_out2
        payoff_2 = strategy_2 * (1 - strategy_2)

        x_mul_att = 1 - torch.sum(payoff_2, dim=1)  # 同样地使用博弈均衡来度量不确定度

        x_max_att = x_max * x_max_att.unsqueeze(1)
        x_mul_att = x_mul * x_mul_att.unsqueeze(1)

        out = self.layer_cat(torch.cat((x_max_att, x_mul_att), dim=1))

        b, c, h, w = out.shape
        out = out.reshape(b, 8, -1, h, w)  # 按照 groups 将通道维度进行重塑
        out = out.permute(0, 2, 1, 3, 4)  # 调整维度顺序，使通道组进行交换
        out = out.reshape(b, -1, h, w)

        return out





class LinearProjection(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64,  bias=True):
        super().__init__()
        inner_dim = dim_head * heads
        self.heads = heads
        self.to_q = nn.Linear(dim, inner_dim, bias = bias)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias = bias)
        self.dim = dim
        self.inner_dim = inner_dim

    def forward(self, x, attn_kv=None):
        B_, N, C = x.shape
        if attn_kv is not None:
            attn_kv = attn_kv.unsqueeze(0).repeat(B_,1,1)
        else:
            attn_kv = x
        N_kv = attn_kv.size(1)
        q = self.to_q(x).reshape(B_, N, 1, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        kv = self.to_kv(attn_kv).reshape(B_, N_kv, 2, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        q = q[0]
        k, v = kv[0], kv[1]
        return q,k,v

"""
def window_partition(x: Tensor, win_size: tuple):
    B, C, H, W = x.shape
    h, w = win_size
    x = x.view(B, C, H // h, h, W // w, w)
    x = x.permute(0, 3, 5, 1, 2, 4).contiguous()
    return x.view(-1, h * w, C)

def window_reverse(x: Tensor, win_size: tuple, H: int, W: int):
    B = x.shape[0] // (H // win_size[0] * W // win_size[1])
    h, w = win_size
    x = x.view(B, H // h, W // w, h * w, -1)
    x = x.permute(0, 3, 1, 4, 2).contiguous()
    return x.view(B, -1, H, W)

class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.2):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        B, C, H, W = x.shape

        # Partition the input into windows
        x_windows = window_partition(x, self.win_size)

        # Perform attention for each window
        B_, N, C = x_windows.shape
        q, k, v = self.qkv(x_windows, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2

        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x_windows = (attn @ v).transpose(1, 2).reshape(B_, N, C)

        # Reverse the window partition
        out = self.proj(x_windows)
        out = self.proj_drop(out)
        out = window_reverse(out, self.win_size, H, W)
        return out
"""

class WindowAttention_sparse(nn.Module):
    def __init__(self, dim, win_size, num_heads, token_projection='linear', qkv_bias=True, qk_scale=None, attn_drop=0.,
                 proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.win_size = win_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * win_size[0] - 1) * (2 * win_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.win_size[0])  # [0,...,Wh-1]
        coords_w = torch.arange(self.win_size[1])  # [0,...,Ww-1]
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.win_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.win_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.win_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        trunc_normal_(self.relative_position_bias_table, std=.02)

        if token_projection == 'linear':
            self.qkv = LinearProjection(dim, num_heads, dim // num_heads, bias=qkv_bias)
        else:
            raise Exception("Projection error!")

        self.token_projection = token_projection
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU()
        self.w = nn.Parameter(torch.ones(2))

    def forward(self, x, attn_kv=None, mask=None):
        c = x.size(1)
        h, w = x.size(2), x.size(3)
        x = x.view(x.size(0), x.size(1), -1).transpose(1, 2)
        B_, N, C = x.shape
        q, k, v = self.qkv(x, attn_kv)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.win_size[0] * self.win_size[1], self.win_size[0] * self.win_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww
        ratio = attn.size(-1) // relative_position_bias.size(-1)
        relative_position_bias = repeat(relative_position_bias, 'nH l c -> nH l (c d)', d=ratio)

        attn = attn + relative_position_bias.unsqueeze(0)

        if mask is not None:
            nW = mask.shape[0]
            mask = repeat(mask, 'nW m n -> nW m (n d)', d=ratio)
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N * ratio) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N * ratio)
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2  # b,h,w,c
        else:
            attn0 = self.softmax(attn)
            attn1 = self.relu(attn) ** 2
            #attn1 = self.relu(attn)
        w1 = torch.exp(self.w[0]) / torch.sum(torch.exp(self.w))
        w2 = torch.exp(self.w[1]) / torch.sum(torch.exp(self.w))
        attn = attn0 * w1 + attn1 * w2
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        out = x.transpose(1, 2).reshape(x.size(0), c, h, w)
        return out

class MSFF_SSA(nn.Module):
    def __init__(self, in_channels_stage1, in_channels_stage2,
                 in_channels_stage3, in_channels_stage4, out_channels):
        super(MSFF_SSA, self).__init__()

        # 膨胀卷积层

        self.dilated_conv1 = nn.Conv2d(in_channels_stage1, out_channels//4, kernel_size=3, padding=1, dilation=1)
        self.dilated_conv2 = nn.Conv2d(in_channels_stage2, out_channels//4, kernel_size=3, padding=2, dilation=2)
        self.dilated_conv3 = nn.Conv2d(in_channels_stage3, out_channels//4, kernel_size=3, padding=4, dilation=4)
        self.dilated_conv4 = nn.Conv2d(in_channels_stage4, out_channels//4, kernel_size=3, padding=8, dilation=8)

        self.ASSA = WindowAttention_sparse(dim=out_channels, win_size=(7, 7), num_heads=8)
        # 门控机制
        self.gate_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, stage1, stage2, stage3, stage4):

        # 经过膨胀卷积处理
        feat1 = self.dilated_conv1(stage1)
        feat2 = self.dilated_conv2(stage2)
        feat3 = self.dilated_conv3(stage3)
        feat4 = self.dilated_conv4(stage4)

        # 下采样特征图
        feat1_up = F.interpolate(feat1, size=feat4.size()[2:], mode='bilinear', align_corners=False)
        feat2_up = F.interpolate(feat2, size=feat4.size()[2:], mode='bilinear', align_corners=False)
        feat3_up = F.interpolate(feat3, size=feat4.size()[2:], mode='bilinear', align_corners=False)

        # 融合所有特征图
        fused_feat = torch.cat((feat1_up, feat2_up, feat3_up, feat4), dim=1)

        out = self.ASSA(fused_feat)
        # 门控机制
        gate = torch.sigmoid(self.gate_conv(fused_feat))
        out = gate * out + fused_feat * (1 - gate)  # 残差连接

        return out


def contrastive_loss(x1, x2, y, margin=1.0):

    #计算对比损失，使用余弦相似度来衡量距离，旨在将x1和x2的特征向量拉近（相同模态）或推远（不同模态）。
    # 计算余弦相似度
    cosine_similarity = F.cosine_similarity(x1, x2)

    # 余弦距离：1 - 余弦相似度
    cosine_distance = 1 - cosine_similarity

    loss = torch.mean((1 - y) * torch.pow(cosine_distance, 2) +
                      y * torch.pow(torch.clamp(margin - cosine_distance, min=0.0), 2))
    return loss


def multimodal_alignment_loss(x_c, x_s, x_f, lambda_align=1.0, margin=1.0):
    """
    MDS策略
    x_c: OCT特征
    x_s: CFP特征
    x_f: 融合后的多模态特征
    """
    # 对比损失：对齐OCT和彩照特征与融合特征
    loss_f_c = contrastive_loss(x_f, x_c, 1, margin)
    loss_f_s = contrastive_loss(x_f, x_s, 1, margin)

    # 总体对齐损失
    alignment_loss = lambda_align * (0.4*loss_f_s + 0.6*loss_f_c)
    return alignment_loss


class OphFusionNet(nn.Module):
    def __init__(self, num_classes, out_channels=1024, conv_dims=(64, 128, 256, 512), model_weight_path="resnet34-333f7ec4.pth"):
        super().__init__()

        self.fc1 = nn.Linear(out_channels, num_classes)
        self.fc2 = nn.Linear(out_channels, num_classes)
        self.fc3 = nn.Linear(out_channels, num_classes)

        self.mcff1 = MSFF_SSA(conv_dims[0], conv_dims[1], conv_dims[2], conv_dims[3], out_channels=out_channels)
        self.mcff2 = MSFF_SSA(conv_dims[0], conv_dims[1], conv_dims[2], conv_dims[3], out_channels=out_channels)
        ###### Hierachical Feature Fusion Block Setting #######

        self.fu = UMF_GTSS(in_channel1=out_channels, in_channel2=out_channels, out_channel=out_channels)
        self.resnet34_oct = resnet34(num_classes=num_classes)
        self.resnet34_fundus = resnet34(num_classes=num_classes)
        pre_weights = torch.load(model_weight_path)
        del_key = []
        for key, _ in pre_weights.items():
             if "fc" in key:
                 del_key.append(key)

        for key in del_key:
             del pre_weights[key]

        self.resnet34_fundus.load_state_dict(pre_weights, strict=False)
        self.resnet34_oct.load_state_dict(pre_weights, strict=False)

    def forward(self, imgs1, imgs2):
        x_s_1, x_s_2, x_s_3, x_s_4, pred1 = self.resnet34_fundus(imgs1)
        x_c_1, x_c_2, x_c_3, x_c_4, pred2 = self.resnet34_oct(imgs2)

        x_s = self.mcff1(x_s_1, x_s_2, x_s_3, x_s_4)
        x_c = self.mcff2(x_c_1, x_c_2, x_c_3, x_c_4)

        x_f = self.fu(x_c, x_s)

        x_f_4_1 = F.adaptive_avg_pool2d(x_f, (1, 1))
        x_c = F.adaptive_avg_pool2d(x_c, (1, 1))
        x_s = F.adaptive_avg_pool2d(x_s, (1, 1))

        # Flatten the tensor
        x_f_4_1 = x_f_4_1.view(x_f_4_1.size(0), -1)  # Flatten to [batch_size, 1024]
        x_c = x_c.view(x_c.size(0), -1)  # Flatten to [batch_size, 1024]
        x_s = x_s.view(x_s.size(0), -1)  # Flatten to [batch_size, 1024]

        align_loss = multimodal_alignment_loss(x_c, x_s, x_f_4_1, 1, 1)

        pred1 = self.fc1(x_c)
        pred2 = self.fc2(x_s)
        pred3 = self.fc3(x_f_4_1)
        return pred3, pred1, pred2, align_loss
        #return x_f_4_1, x_c, x_s, align_loss
