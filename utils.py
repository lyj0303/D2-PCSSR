import random
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np
import random


def gaussian_kernel(x, bandwidth=1.0):
    """
    高斯核函数
    """
    return torch.exp(-0.5 * (x / bandwidth) ** 2) / (bandwidth * torch.sqrt(torch.tensor(2 * torch.pi)))


def compute_mutual_information(x, y, bandwidth=1.0):
    """
    计算两个向量 x 和 y 之间的互信息，使用核密度估计（KDE）来替代直方图。

    参数：
        x, y: 输入的两个向量
        bandwidth: 高斯核的带宽，控制平滑程度

    返回：
        互信息值
    """
    # 确保输入的张量需要梯度
    x = x.requires_grad_(True)
    y = y.requires_grad_(True)

    # 核密度估计
    n = x.size(0)

    # 计算 X 和 Y 的核密度
    diff_x = x.unsqueeze(1) - x.unsqueeze(0)
    diff_y = y.unsqueeze(1) - y.unsqueeze(0)

    K_x = gaussian_kernel(diff_x)
    K_y = gaussian_kernel(diff_y)

    # 计算边际分布
    p_x = K_x.sum(dim=1) / n
    p_y = K_y.sum(dim=1) / n

    # 计算联合分布
    K_xy = gaussian_kernel(diff_x + diff_y)
    joint_prob = K_xy / n

    # 计算熵
    H_x = -torch.sum(p_x * torch.log(p_x + 1e-10))
    H_y = -torch.sum(p_y * torch.log(p_y + 1e-10))
    H_xy = -torch.sum(joint_prob * torch.log(joint_prob + 1e-10))

    # 互信息 = H(x) + H(y) - H(x, y)
    MI = H_x + H_y - H_xy
    return MI