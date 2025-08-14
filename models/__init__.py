"""
YOLO-MHAF模型包
实现基于改进YOLO v8n的多头自适应融合检测算法
"""

from .yolo_mhaf import YOLOMHAF
from .modules import *

__all__ = ['YOLOMHAF']