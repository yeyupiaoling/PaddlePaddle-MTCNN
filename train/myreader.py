import mmap
from multiprocessing import cpu_count

import cv2
import numpy as np
import paddle


class ImageData(object):
    def __init__(self, data_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        # 获取label
        self.label = {}
        for line in open(data_path + '.label', 'rb'):
            key, label = line.split('\t'.encode('ascii'))
            self.label[key] = label
        # 获取box
        self.box = {}
        for line in open(data_path + '.box', 'rb'):
            key, box0, box1, box2, box3 = line.split('\t'.encode('ascii'))
            self.box[key] = [float(box0), float(box1), float(box2), float(box3)]
        # 获取landmark
        self.landmark = {}
        for line in open(data_path + '.landmark', 'rb'):
            key, landmark0, landmark1, landmark2, landmark3, landmark4, landmark5, landmark6, landmark7, landmark8, \
            landmark9 = line.split('\t'.encode('ascii'))
            self.landmark[key] = [float(landmark0), float(landmark1), float(landmark2), float(landmark3),
                                  float(landmark4), float(landmark5), float(landmark6), float(landmark7),
                                  float(landmark8), float(landmark9), ]

    # 获取图像数据
    def get_img(self, key):
        p = self.offset_dict.get(key, None)
        if p is None:
            return None
        val_pos, val_len = p
        return self.m[val_pos:val_pos + val_len]

    # 获取图像标签
    def get_label(self, key):
        return self.label.get(key)

    # 获取人脸box
    def get_bbox(self, key):
        return self.box.get(key)

    # 获取关键点
    def get_landmark(self, key):
        return self.landmark.get(key)

    # 获取所有keys
    def get_keys(self):
        return self.offset_dict.keys()


def train_mapper(sample):
    image, label, bbox, landmark = sample
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, True)
    # 把图片转换成numpy值
    image = np.array(image).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 转换成BGR
    image = image[(2, 1, 0), :, :] / 255.0
    return image, [int(label)], bbox, landmark


# 获取训练的reader
def train_reader(data_path):
    def reader():
        imageData = ImageData(data_path)
        keys = imageData.get_keys()
        keys = list(keys)
        np.random.shuffle(keys)
        for key in keys:
            img = imageData.get_img(key)
            label = imageData.get_label(key)
            bbox = imageData.get_bbox(key)
            landmark = imageData.get_landmark(key)
            yield img, label, bbox, landmark

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), 102400)


if __name__ == '__main__':
    t_reader = paddle.batch(reader=train_reader('../data/12/all_data'), batch_size=32)
    for data in t_reader():
        print(data)
