"""
YOLO-MHAF主模型
基于改进YOLO v8n的多头自适应融合检测算法
整合Ghost模块、SPD-Conv、混合注意力机制、C2f-Hybrid和AMFN等组件
"""

import torch
import torch.nn as nn
import math
from typing import List, Optional, Tuple

from .modules.ghost import GhostConv, C2f_Ghost
from .modules.spd_conv import SPDConv, C2f_SPD
from .modules.attention import HybridAttention
from .modules.c2f_hybrid import C2f_Hybrid, SPPF_Hybrid
from .modules.amfn import AMFN, AMFN_Lite


class Conv(nn.Module):
    """标准卷积层"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2f(nn.Module):
    """标准C2f模块"""
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)
        self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g, k=((3, 3)), e=1.0) for _ in range(n))

    def forward(self, x):
        y = list(self.cv1(x).chunk(2, 1))
        y.extend(m(y[-1]) for m in self.m)
        return self.cv2(torch.cat(y, 1))


class Bottleneck(nn.Module):
    """标准瓶颈模块"""
    def __init__(self, c1, c2, shortcut=True, g=1, k=(3, 3), e=0.5):
        super().__init__()
        c_ = int(c2 * e)
        self.cv1 = Conv(c1, c_, k[0], 1)
        self.cv2 = Conv(c_, c2, k[1], 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class SPPF(nn.Module):
    """标准SPPF模块"""
    def __init__(self, c1, c2, k=5):
        super().__init__()
        c_ = c1 // 2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        y1 = self.m(x)
        y2 = self.m(y1)
        return self.cv2(torch.cat((x, y1, y2, self.m(y2)), 1))


class Detect(nn.Module):
    """YOLO检测头"""
    def __init__(self, nc=80, ch=()):
        super().__init__()
        self.nc = nc  # 类别数
        self.no = nc + 5  # 每个anchor的输出数
        self.nl = len(ch)  # 检测层数
        self.na = 1  # anchor数量
        self.grid = [torch.zeros(1)] * self.nl
        self.anchor_grid = [torch.zeros(1)] * self.nl
        self.m = nn.ModuleList(nn.Conv2d(x, self.no * self.na, 1) for x in ch)

    def forward(self, x):
        z = []
        for i in range(self.nl):
            x[i] = self.m[i](x[i])
            bs, _, ny, nx = x[i].shape
            x[i] = x[i].view(bs, self.na, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()
            z.append(x[i])
        return x if self.training else (torch.cat(z, 1),)


class YOLOMHAF(nn.Module):
    """YOLO-MHAF主模型
    基于改进YOLO v8n的多头自适应融合检测算法
    """
    
    def __init__(self, cfg=None, ch=3, nc=None, anchors=None):
        super().__init__()
        if isinstance(cfg, dict):
            self.yaml = cfg
        else:
            import yaml
            self.yaml_file = Path(cfg).name
            with open(cfg, encoding='ascii', errors='ignore') as f:
                self.yaml = yaml.safe_load(f)

        # 定义模型
        ch = self.yaml['ch'] = self.yaml.get('ch', ch)
        if nc and nc != self.yaml['nc']:
            self.yaml['nc'] = nc
        if anchors:
            self.yaml['anchors'] = round(anchors)
        self.model, self.save = self.parse_model(deepcopy(self.yaml), ch=[ch])
        self.names = [str(i) for i in range(self.yaml['nc'])]
        self.inplace = self.yaml.get('inplace', True)

        # 构建步长
        m = self.model[-1]
        if isinstance(m, Detect):
            s = 256
            m.inplace = self.inplace
            forward = lambda x: self.forward(x)[0] if isinstance(m, Detect) else self.forward(x)
            m.stride = torch.tensor([s / x.shape[-2] for x in forward(torch.zeros(1, ch, s, s))])
            m.anchors /= m.stride.view(-1, 1, 1)
            check_anchor_order(m)
            self.stride = m.stride
            self._initialize_biases()

    def forward(self, x, augment=False, profile=False, visualize=False):
        if augment:
            return self._forward_augment(x)
        return self._forward_once(x, profile, visualize)

    def _forward_augment(self, x):
        img_size = x.shape[-2:]
        s = [1, 0.83, 0.67]
        f = [None, 3, None]
        y = []
        for si, fi in zip(s, f):
            xi = scale_img(x.flip(fi) if fi else x, si, gs=int(self.stride.max()))
            yi = self._forward_once(xi)[0]
            yi = self._descale_pred(yi, fi, si, img_size)
            y.append(yi)
        y = self._clip_augmented(y)
        return torch.cat(y, 1), None

    def _forward_once(self, x, profile=False, visualize=False):
        y, dt = [], []
        for m in self.model:
            if m.f != -1:
                x = y[m.f] if isinstance(m.f, int) else [x if j == -1 else y[j] for j in m.f]
            if profile:
                self._profile_one_layer(m, x, dt)
            x = m(x)
            y.append(x if m.i in self.save else None)
            if visualize:
                feature_visualization(x, m.type, m.i, save_dir=visualize)
        return x

    def _initialize_biases(self, cf=None):
        """初始化Detect()偏置，WARNING: 需要stride可用"""
        m = self.model[-1]
        for mi, s in zip(m.m, m.stride):
            b = mi.bias.view(m.na, -1)
            b.data[:, 4] += math.log(8 / (640 / s) ** 2)
            b.data[:, 5:5 + m.nc] += math.log(0.6 / (m.nc - 0.999999)) if cf is None else torch.log(cf / cf.sum())
            mi.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def parse_model(self, d, ch):
        """解析模型配置"""
        anchors, nc, gd, gw = d['anchors'], d['nc'], d['depth_multiple'], d['width_multiple']
        na = (len(anchors[0]) // 2) if isinstance(anchors, list) else anchors
        no = na * (nc + 5)

        layers, save, c2 = [], [], ch[-1]
        for i, (f, n, m, args) in enumerate(d['backbone'] + d['head']):
            m = eval(m) if isinstance(m, str) else m
            for j, a in enumerate(args):
                with contextlib.suppress(NameError):
                    args[j] = eval(a) if isinstance(a, str) else a

            n = n_ = max(round(n * gd), 1) if n > 1 else n
            if m in {Conv, GhostConv, Bottleneck, SPDConv, C2f, C2f_Ghost, C2f_SPD, C2f_Hybrid, SPPF, SPPF_Hybrid}:
                c1, c2 = ch[f], args[0]
                if c2 != no:
                    c2 = make_divisible(c2 * gw, 8)

                args = [c1, c2, *args[1:]]
                if m in {C2f, C2f_Ghost, C2f_SPD, C2f_Hybrid}:
                    args.insert(2, n)
                    n = 1
            elif m is nn.BatchNorm2d:
                args = [ch[f]]
            elif m is Concat:
                c2 = sum(ch[x] for x in f)
            elif m in {Detect}:
                args.append([ch[x] for x in f])
                if isinstance(args[1], int):
                    args[1] = [list(range(args[1] * 2))] * len(f)
            elif m is AMFN:
                c1 = [ch[x] for x in f]
                c2 = args[0]
                args = [c1, c2, *args[1:]]
            else:
                c2 = ch[f]

            m_ = nn.Sequential(*(m(*args) for _ in range(n))) if n > 1 else m(*args)
            t = str(m)[8:-2].replace('__main__.', '')
            np = sum(x.numel() for x in m_.parameters())
            m_.i, m_.f, m_.type, m_.np = i, f, t, np
            save.extend(x % i for x in ([f] if isinstance(f, int) else f) if x != -1)
            layers.append(m_)
            if i == 0:
                ch = []
            ch.append(c2)
        return nn.Sequential(*layers), sorted(save)


def make_divisible(x, divisor):
    """返回能被divisor整除的最接近x的数"""
    if isinstance(divisor, torch.Tensor):
        divisor = int(divisor.max())
    return math.ceil(x / divisor) * divisor


class Concat(nn.Module):
    """连接操作"""
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


def check_anchor_order(m):
    """检查anchor顺序vs stride顺序"""
    a = m.anchors.prod(-1).mean(-1).view(-1)
    da = a[-1] - a[0]
    ds = m.stride[-1] - m.stride[0]
    if da and (da.sign() != ds.sign()):
        m.anchors[:] = m.anchors.flip(0)


# 模型配置
def create_yolo_mhaf_config():
    """创建YOLO-MHAF模型配置"""
    return {
        'nc': 5,  # 花生叶片病害类别数（假设5类）
        'depth_multiple': 0.33,
        'width_multiple': 0.25,
        'ch': 3,
        
        # 骨干网络 - 混合轻量化策略
        'backbone': [
            [-1, 1, Conv, [64, 6, 2, 2]],  # 0-P1/2
            [-1, 1, Conv, [128, 3, 2]],    # 1-P2/4
            [-1, 3, C2f_Ghost, [128, True]],  # 2 - 浅层使用Ghost模块
            [-1, 1, Conv, [256, 3, 2]],    # 3-P3/8
            [-1, 6, C2f_Ghost, [256, True]],  # 4 - 浅层使用Ghost模块
            [-1, 1, SPDConv, [512, 3, 2]],  # 5-P4/16 - 深层使用SPD-Conv
            [-1, 6, C2f_SPD, [512, True]],   # 6 - 深层使用SPD-Conv
            [-1, 1, SPDConv, [1024, 3, 2]], # 7-P5/32 - 深层使用SPD-Conv
            [-1, 3, C2f_SPD, [1024, True]], # 8 - 深层使用SPD-Conv
            [-1, 1, SPPF_Hybrid, [1024, 5]], # 9 - 带混合注意力的SPPF
        ],
        
        # 检测头 - 自适应多路径特征融合网络
        'head': [
            [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 10
            [[-1, 6], 1, Concat, [1]], # 11 cat backbone P4
            [-1, 3, C2f_Hybrid, [512, False, 1, 0.5, True, 'hybrid', 'global']], # 12 - C2f-Hybrid
            
            [-1, 1, nn.Upsample, [None, 2, 'nearest']], # 13
            [[-1, 4], 1, Concat, [1]], # 14 cat backbone P3
            [-1, 3, C2f_Hybrid, [256, False, 1, 0.5, True, 'hybrid', 'global']], # 15 - C2f-Hybrid (P3/8-small)

            [-1, 1, Conv, [256, 3, 2]], # 16
            [[-1, 12], 1, Concat, [1]], # 17 cat head P4
            [-1, 3, C2f_Hybrid, [512, False, 1, 0.5, True, 'hybrid', 'global']], # 18 - C2f-Hybrid (P4/16-medium)

            [-1, 1, Conv, [512, 3, 2]], # 19
            [[-1, 9], 1, Concat, [1]], # 20 cat head P5
            [-1, 3, C2f_Hybrid, [1024, False, 1, 0.5, True, 'hybrid', 'global']], # 21 - C2f-Hybrid (P5/32-large)

            # AMFN特征融合
            [[15, 18, 21], 1, AMFN, [256, 3, 2, True, 2]], # 22 - AMFN融合

            # 检测头
            [[15, 18, 21], 1, Detect, [5]], # 23 Detect(P3, P4, P5)
        ]
    }


def yolo_mhaf_n(pretrained=False, **kwargs):
    """创建YOLO-MHAF-n模型"""
    cfg = create_yolo_mhaf_config()
    model = YOLOMHAF(cfg, **kwargs)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def yolo_mhaf_s(pretrained=False, **kwargs):
    """创建YOLO-MHAF-s模型"""
    cfg = create_yolo_mhaf_config()
    cfg['depth_multiple'] = 0.33
    cfg['width_multiple'] = 0.50
    model = YOLOMHAF(cfg, **kwargs)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


def yolo_mhaf_m(pretrained=False, **kwargs):
    """创建YOLO-MHAF-m模型"""
    cfg = create_yolo_mhaf_config()
    cfg['depth_multiple'] = 0.67
    cfg['width_multiple'] = 0.75
    model = YOLOMHAF(cfg, **kwargs)
    if pretrained:
        # TODO: 加载预训练权重
        pass
    return model


if __name__ == '__main__':
    # 测试模型
    model = yolo_mhaf_n()
    x = torch.randn(1, 3, 640, 640)
    y = model(x)
    print(f'Input shape: {x.shape}')
    print(f'Output shape: {[yi.shape for yi in y] if isinstance(y, (list, tuple)) else y.shape}')
    
    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params:,}')