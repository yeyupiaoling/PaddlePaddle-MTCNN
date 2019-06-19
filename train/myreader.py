import mmap

import cv2
import numpy as np


class ImageData(object):
    def __init__(self, data_path, label_path):
        self.offset_dict = {}
        for line in open(data_path + '.header', 'rb'):
            key, val_pos, val_len = line.split('\t'.encode('ascii'))
            self.offset_dict[key] = (int(val_pos), int(val_len))
        self.fp = open(data_path + '.data', 'rb')
        self.m = mmap.mmap(self.fp.fileno(), 0, access=mmap.ACCESS_READ)
        print('loading label')
        # 获取label
        self.label = {}
        self.box = {}
        self.landmark = {}
        if not label_path:
            label_path = data_path + '.label'
        for line in open(label_path, 'rb'):
            key, bbox, landmark, label = line.split(b'\t')
            self.label[key] = int(label)
            self.box[key] = [float(x) for x in bbox.split()]
            self.landmark[key] = [float(x) for x in landmark.split()]
        print('finish loading data:', len(self.label))

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
        return self.label.keys()


def train_mapper(sample):
    image, label, bbox, landmark = sample
    image = np.fromstring(image, dtype=np.uint8)
    image = cv2.imdecode(image, 1)
    assert (image is not None), 'image is None'
    # 把图片转换成numpy值
    image = np.array(image).astype(np.float32)
    # 转换成CHW
    image = image.transpose((2, 0, 1))
    # 转换成BGR
    image = image[(2, 1, 0), :, :] / 255.0
    return image, [int(label)], bbox, landmark


def train_reader(data_path, label_path, batch_size):
    def reader():
        imageData = ImageData(data_path, label_path)
        keys = imageData.get_keys()
        keys = list(keys)
        np.random.shuffle(keys)

        batch_img, batch_label, batch_bbox, batch_landmark = [], [], [], []
        for key in keys:
            img = imageData.get_img(key)
            assert (img is not None)
            label = imageData.get_label(key)
            assert (label is not None)
            bbox = imageData.get_bbox(key)
            landmark = imageData.get_landmark(key)
            sample = (img, label, bbox, landmark)
            img, label, bbox, landmark = train_mapper(sample)
            # # reshape
            img = img.reshape([1] + list(img.shape))
            label = np.array(label, np.int64).reshape([1, -1])
            bbox = np.array(bbox, np.float32).reshape([1, -1])
            landmark = np.array(landmark, np.float32).reshape([1, -1])
            batch_img.append(img)
            batch_label.append(label)
            batch_bbox.append(bbox)
            batch_landmark.append(landmark)
            if len(batch_img) == batch_size:
                yield np.vstack(batch_img), np.vstack(batch_label), np.vstack(batch_bbox), np.vstack(batch_landmark)
                batch_img, batch_label, batch_bbox, batch_landmark = [], [], [], []

    return reader
