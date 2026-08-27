import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np
import torch.nn.functional as F
import scipy.io as sio
import copy
from torch.utils.data.dataset import T_co
from torchvision import transforms


def random_rotation_numpy(images):
    """
    对一批多通道图像进行随机旋转，旋转角度为 90°, 180°, 270°, 360°。

    Args:
        images (np.ndarray): 输入图像，形状为 (batch_size, channels, height, width)。

    Returns:
        np.ndarray: 随机旋转后的图像，形状与输入相同。
    """
    # 确保输入是四维数组
    assert images.ndim == 3, "Input must have shape (height, width, channels)"

    # 随机选择旋转角度
    rotation_angles = [0, 90, 180, 270]  # 360° 与 0° 相同
    angle = random.choice(rotation_angles)

    if angle == 90:
        # 逆时针旋转 90°
        rotated_images = np.rot90(images, k=1, axes=(0, 1))
    elif angle == 180:
        # 逆时针旋转 180°
        rotated_images = np.rot90(images, k=2, axes=(0, 1))
    elif angle == 270:
        # 逆时针旋转 270°
        rotated_images = np.rot90(images, k=3, axes=(0, 1))
    else:
        # 不旋转
        rotated_images = images

    return rotated_images

def min_max_normalize(data):
    h, w, C = data.shape
    for c in range(C):
        Max = np.max(data[:, :, c])
        Min = np.min(data[:, :, c])
        if Max - Min != 0:
            data[:, :, c] = (data[:, :, c] - Min) / (Max - Min)
        else:
            data[:, :, c] = 0


    return data

class patch_dataset(Dataset):
    def __init__(self, train=True) -> None:
        super().__init__()
        self.images = []
        self.train = train
        # indexes = sio.loadmat('../data/SAR_DATA/index.mat')['index'].tolist()
        indexes = []
        with open('../coordinates_finnal.txt', 'r') as file:
            # 遍历每一行
            for line in file:
                # 去掉换行符并以逗号分隔x和y
                x, y = line.strip().split(',')
                # 将x和y转换为整数（或float），然后加入列表
                indexes.append((int(x), int(y)))
        data1 = sio.loadmat('../data/SAR_DATA/new_data/0510/c3.mat')['data']
        data2 = sio.loadmat('../data/SAR_DATA/new_data/0510/freeman.mat')['data']
        data3 = sio.loadmat('../data/SAR_DATA/new_data/0510/h_a_alpha.mat')['data']
        data4 = sio.loadmat('../data/SAR_DATA/new_data/0510/nned.mat')['data']
        data5 = sio.loadmat('../data/SAR_DATA/new_data/0510/tsvm.mat')['data']
        data6 = sio.loadmat('../data/SAR_DATA/new_data/0510/yamaguchi.mat')['data']
        #t9 = sio.loadmat('../data/SAR_DATA/0603/t9.mat')['data2']
        data = np.concatenate((data6, data5, data4, data3, data2, data1), axis=2)[:, :, [5, 6, 9, 13, 17, 18]]
        #计算每个通道的均值和标准差（这里假设是 3 通道数据）
        # h, w, D = data.shape
        # for d in range(D):
        #     avg = np.mean(data[:, :, d])
        #     std = np.std(data[:, :, d])
        #     if std != 0:
        #         data[:, :, d] = (data[:, :, d] - avg) / std
        #     else:
        #         data[:, :, d] = 0
        data = min_max_normalize(data)
        for (x, y) in indexes:
            x = x
            y = y
            print(x, y)
            patch = np.concatenate((data1[x-5:x+6, y-5:y+6, :], data2[x-5:x+6, y-5:y+6, :], data3[x-5:x+6, y-5:y+6, :],
                                   data4[x-5:x+6, y-5:y+6, :], data5[x-5:x+6, y-5:y+6, :], data6[x-5:x+6, y-5:y+6, :]),
                                   axis=2)
            patch = np.asarray(patch)


            #patch = data[x-2:x+3, y-2:y+3, :]

            patch = data[x, y, :]

            print(patch.shape)
            self.images.append(patch)
        self.images = np.asarray(self.images)
        print(self.images.shape)
        # for c in range(self.images.shape[3]):
        #     channel_data = self.images[:, :, :, c]  # 提取当前通道数据
        #     min_val = channel_data.min()  # 当前通道的最小值
        #     max_val = channel_data.max()  # 当前通道的最大值
        #     self.images[:, :, :, c] = (channel_data - min_val) / (max_val - min_val)
        self.total_labels = sio.loadmat('../data/gt.mat')['gt'][35:70, 1]
        Max = np.max(self.total_labels)
        Min = np.min(self.total_labels)
        # self.total_labels = (self.total_labels - Min) / ((Max - Min) + 1e-10)
        self.total_images = np.asarray(self.images)
        self.images = copy.deepcopy(self.total_images)
        self.labels = copy.deepcopy(self.total_labels)
        print("images.shape", self.images.shape)
        print("label.shape:", self.labels.shape)

    def __getitem__(self, index):
        if self.train:
            # img = random_rotation_numpy(self.images[index])
            # img = img.copy()
            img = self.images[index]
        else:
            img = self.images[index]
        # return torch.tensor(img, dtype=torch.float32).permute(2, 0, 1), \
        #     torch.tensor(self.labels[index], dtype=torch.float32)

        return torch.tensor(img, dtype=torch.float32), \
            torch.tensor(self.labels[index], dtype=torch.float32)

    def __len__(self):
        return self.labels.shape[0]

    def rebulid(self, data_indexex):
        self.labels = copy.deepcopy(self.total_labels[data_indexex])
        self.images = copy.deepcopy(self.total_images[data_indexex])



# dataset = patch_dataset(data_indexes=list(range(10)))
# dataloader = DataLoader(dataset=dataset, drop_last=True, batch_size=1)
# for (image, label) in dataloader:
#     print(image.shape, label.shape)
#     print(label)
#     print("--------------------------------")

