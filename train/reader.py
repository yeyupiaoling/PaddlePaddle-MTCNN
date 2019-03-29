from multiprocessing import cpu_count

import numpy as np
import paddle
from PIL import Image


def train_mapper(sample):
    sample = sample.split(' ')
    image = sample[0]
    label = [int(sample[1])]
    # 做补0预操作
    bbox = [0, 0, 0, 0]
    landmark = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]

    # 如果只有box，那天关键点就补0
    if len(sample) == 6:
        bbox = [float(i) for i in sample[2:]]

    # 如果只有关键点，那么box就补0
    if len(sample) == 12:
        landmark = [float(i) for i in sample[2:]]

    image = Image.open(image)
    # 把图片转换成numpy值
    image = np.array(image).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 转换成BGR
    image = image[(2, 1, 0), :, :] / 255.0
    return image, label, bbox, landmark


# 获取训练的reader
def train_reader(data_list_path):
    def reader():
        with open(data_list_path, 'r') as f:
            lines = f.readlines()
        np.random.shuffle(lines)
        for line in lines:
            yield line

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)
