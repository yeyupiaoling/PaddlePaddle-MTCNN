import os
import pickle
import random

import cv2
import numpy as np
from tqdm import tqdm

from utils import IOU, crop_landmark_image, combine_data_list


# 截取pos,neg,part三种类型图片并resize成24x24大小作为RNet的输入
def crop_24_box_image(data_path):
    pass





if __name__ == '__main__':
    data_path = '../data/'
    # 获取人脸的box图片数据
    crop_24_box_image(data_path)
    # 获取人脸关键点的数据
    crop_landmark_image(data_path, 24, argument=True)
    # 合并数据列表
    combine_data_list(os.path.join(data_path, '24'))
