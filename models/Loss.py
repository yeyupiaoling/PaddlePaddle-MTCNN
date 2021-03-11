import paddle.nn as nn
import paddle
import numpy as np


class ClassLoss(nn.Layer):
    def __init__(self):
        super(ClassLoss, self).__init__(name_scope='ClassLoss')
        self.entropy_loss = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
        self.keep_ratio = 0.7

    def forward(self, class_out, label):
        # 保留neg 0 和pos 1 的数据，忽略掉part -1, landmark -2
        zeros = paddle.zeros_like(label)
        ignore_label = paddle.full_like(label, fill_value=-100)
        label = paddle.where(paddle.less_than(label, zeros), ignore_label, label)
        # 求neg 0 和pos 1 的数据70%数据
        ones = paddle.ones_like(label)
        valid_label = paddle.where(paddle.greater_equal(label, zeros), ones, zeros)
        num_valid = paddle.sum(valid_label)
        keep_num = int((num_valid * self.keep_ratio).numpy()[0])
        # 计算交叉熵损失
        loss = self.entropy_loss(input=class_out, label=label)
        # 取有效数据的70%计算损失
        loss, _ = paddle.topk(paddle.squeeze(loss), k=keep_num)
        return paddle.mean(loss)


class BBoxLoss(nn.Layer):
    def __init__(self):
        super(BBoxLoss, self).__init__(name_scope='BBoxLoss')
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, bbox_out, bbox_target, label):
        # 保留pos 1 和part -1 的数据
        ones = paddle.ones_like(label)
        zeros = paddle.zeros_like(label)
        valid_label = paddle.where(paddle.equal(paddle.abs(label), ones), ones, zeros)
        # 获取有效值的总数的70%
        keep_num = int(paddle.sum(valid_label).numpy()[0] * self.keep_ratio)
        loss = self.square_loss(input=bbox_out, label=bbox_target)
        loss = loss * valid_label
        # 取有效数据的70%计算损失
        _, index = paddle.topk(paddle.sum(loss, axis=1), k=keep_num, axis=0)
        loss = paddle.gather(loss, index)
        return paddle.mean(loss)


class LandmarkLoss(nn.Layer):
    def __init__(self):
        super(LandmarkLoss, self).__init__(name_scope='LandmarkLoss')
        self.square_loss = nn.MSELoss(reduction='none')
        self.keep_ratio = 1.0

    def forward(self, landmark_out, landmark_target, label):
        # 只保留landmark数据 -2
        ones = paddle.ones_like(label)
        zeros = paddle.zeros_like(label)
        valid_label = paddle.where(paddle.equal(label, paddle.full_like(label, fill_value=-2)), ones, zeros)
        # 获取有效值的总数的70%
        keep_num = int(paddle.sum(valid_label).numpy()[0] * self.keep_ratio)
        loss = self.square_loss(input=landmark_out, label=landmark_target)
        loss = loss * valid_label
        # 取有效数据的70%计算损失
        _, index = paddle.topk(paddle.sum(loss, axis=1), k=keep_num, axis=0)
        loss = paddle.gather(loss, index)
        return paddle.mean(loss)


# 求训练时的准确率
def accuracy(class_out, label):
    # 查找neg 0 和pos 1所在的位置
    zeros = paddle.zeros_like(label)
    cond = paddle.greater_equal(label, zeros)
    picked, _ = np.where(cond.numpy())
    picked = paddle.to_tensor(picked, dtype='int32')
    # 求neg 0 和pos 1的准确率
    valid_class_out = paddle.gather(class_out, picked)
    valid_label = paddle.gather(label, picked)
    acc = paddle.metric.accuracy(valid_class_out, valid_label)
    return acc
