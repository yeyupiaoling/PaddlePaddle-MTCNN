import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.regularizer import L2DecayRegularizer
import numpy as np


def P_Net():
    # 定义输入层
    image = fluid.layers.data(name='image', shape=[3, 12, 12], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    bbox_target = fluid.layers.data(name='bbox_target', shape=[4], dtype='float32')
    landmark_target = fluid.layers.data(name='landmark_target', shape=[10], dtype='float32')

    # 第一层卷积层
    conv1 = fluid.layers.conv2d(input=image,
                                num_filters=10,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv1')
    conv1_prelu = fluid.layers.prelu(x=conv1, mode='all', name='conv1_prelu')

    # 唯一一个池化层
    pool1 = fluid.layers.pool2d(input=conv1_prelu,
                                pool_size=2,
                                pool_stride=2,
                                name='pool1')

    # 第二层卷积层
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=16,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv2')
    conv2_prelu = fluid.layers.prelu(x=conv2, mode='all', name='conv2_prelu')

    # 第三层卷积层
    conv3 = fluid.layers.conv2d(input=conv2_prelu,
                                num_filters=32,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv3')
    conv3_prelu = fluid.layers.prelu(x=conv3, mode='all', name='conv3_prelu')

    # 分类是否人脸的卷积输出层
    conv4_1 = fluid.layers.conv2d(input=conv3_prelu,
                                  num_filters=2,
                                  filter_size=1,
                                  param_attr=ParamAttr(initializer=Xavier(),
                                                       regularizer=L2DecayRegularizer(0.0005)),
                                  name='conv4_1')
    conv4_1 = fluid.layers.squeeze(input=conv4_1, axes=[])
    conv4_1_softmax = fluid.layers.softmax(input=conv4_1, use_cudnn=False)

    # 人脸box的回归卷积输出层
    conv4_2 = fluid.layers.conv2d(input=conv3_prelu,
                                  num_filters=4,
                                  filter_size=1,
                                  param_attr=ParamAttr(initializer=Xavier(),
                                                       regularizer=L2DecayRegularizer(0.0005)),
                                  name='conv4_2')

    # 5个关键点的回归卷积输出层
    conv4_3 = fluid.layers.conv2d(input=conv3_prelu,
                                  num_filters=10,
                                  filter_size=1,
                                  param_attr=ParamAttr(initializer=Xavier(),
                                                       regularizer=L2DecayRegularizer(0.0005)),
                                  name='conv4_3')

    # 获取是否人脸分类交叉熵损失函数
    cls_prob = fluid.layers.squeeze(input=conv4_1_softmax, axes=[], name='cls_prob')
    label_cost = cls_ohem(cls_prob=cls_prob, label=label)

    # 获取人脸box回归平方差损失函数
    bbox_pred = fluid.layers.squeeze(input=conv4_2, axes=[], name='bbox_pred')
    bbox_loss = bbox_ohem(bbox_pred=bbox_pred, bbox_target=bbox_target, label=label)

    # 获取人脸5个关键点回归平方差损失函数
    landmark_pred = fluid.layers.squeeze(input=conv4_3, axes=[], name='landmark_pred')
    landmark_loss = landmark_ohem(landmark_pred=landmark_pred, landmark_target=landmark_target, label=label)

    # 准确率函数
    accuracy = cal_accuracy(cls_prob=cls_prob, label=label)
    return image, label, bbox_target, landmark_target, label_cost, bbox_loss, landmark_loss, accuracy, conv4_1, conv4_2, conv4_3


def R_Net():
    # 定义输入层
    image = fluid.layers.data(name='image', shape=[3, 24, 24], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    bbox_target = fluid.layers.data(name='bbox_target', shape=[4], dtype='float32')
    landmark_target = fluid.layers.data(name='landmark_target', shape=[10], dtype='float32')

    # 第一层卷积层
    conv1 = fluid.layers.conv2d(input=image,
                                num_filters=28,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv1')
    conv1_prelu = fluid.layers.prelu(x=conv1, mode='all', name='conv1_prelu')

    # 第一个池化层
    pool1 = fluid.layers.pool2d(input=conv1_prelu,
                                pool_size=3,
                                pool_stride=2,
                                name='pool1')

    # 第二层卷积层
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=48,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv2')
    conv2_prelu = fluid.layers.prelu(x=conv2, mode='all', name='conv2_prelu')

    # 第二个池化层
    pool2 = fluid.layers.pool2d(input=conv2_prelu,
                                pool_size=3,
                                pool_stride=2,
                                name='pool2')

    # 第三层卷积层
    conv3 = fluid.layers.conv2d(input=pool2,
                                num_filters=64,
                                filter_size=2,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv3')
    conv3_prelu = fluid.layers.prelu(x=conv3, mode='all', name='conv3_prelu')

    # 把图像特征进行展开
    fc_flatten = fluid.layers.flatten(conv3_prelu)

    # 第一个全连接层
    fc1 = fluid.layers.fc(input=fc_flatten, size=128, name='fc1')

    # 是否人脸的分类输出层
    cls_prob = fluid.layers.fc(input=fc1, size=2, act='softmax', name='cls_fc')
    # 是否人脸分类输出交叉熵损失函数
    cls_loss = cls_ohem(cls_prob=cls_prob, label=label)

    # 人脸box的输出层
    bbox_pred = fluid.layers.fc(input=fc1, size=4, act=None, name='bbox_fc')
    # 人脸box的平方差损失函数
    bbox_loss = bbox_ohem(bbox_pred=bbox_pred, bbox_target=bbox_target, label=label)

    # 人脸5个关键点的输出层
    landmark_pred = fluid.layers.fc(input=fc1, size=10, act=None, name='landmark_fc')
    # 人脸关键点的平方差损失函数
    landmark_loss = landmark_ohem(landmark_pred=landmark_pred, landmark_target=landmark_target, label=label)

    # 准确率函数
    accuracy = cal_accuracy(cls_prob=cls_prob, label=label)
    return image, label, bbox_target, landmark_target, cls_loss, bbox_loss, landmark_loss, accuracy, cls_prob, bbox_pred, landmark_pred


def O_Net():
    # 定义输入层
    image = fluid.layers.data(name='image', shape=[3, 48, 48], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')
    bbox_target = fluid.layers.data(name='bbox_target', shape=[4], dtype='float32')
    landmark_target = fluid.layers.data(name='landmark_target', shape=[10], dtype='float32')

    # 第一层卷积层
    conv1 = fluid.layers.conv2d(input=image,
                                num_filters=32,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv1')
    conv1_prelu = fluid.layers.prelu(x=conv1, mode='all', name='conv1_prelu')

    # 第一个池化层
    pool1 = fluid.layers.pool2d(input=conv1_prelu,
                                pool_size=3,
                                pool_stride=2,
                                name='pool1')

    # 第二层卷积层
    conv2 = fluid.layers.conv2d(input=pool1,
                                num_filters=64,
                                filter_size=3,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv2')
    conv2_prelu = fluid.layers.prelu(x=conv2, mode='all', name='conv2_prelu')

    # 第二个池化层
    pool2 = fluid.layers.pool2d(input=conv2_prelu,
                                pool_size=3,
                                pool_stride=2,
                                name='pool2')

    # 第三层卷积层
    conv3 = fluid.layers.conv2d(input=pool2,
                                num_filters=64,
                                filter_size=2,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv3')
    conv3_prelu = fluid.layers.prelu(x=conv3, mode='all', name='conv3_prelu')

    # 第三个池化层
    pool3 = fluid.layers.pool2d(input=conv3_prelu,
                                pool_size=2,
                                pool_stride=2,
                                name='pool3')

    # 第四层卷积层
    conv4 = fluid.layers.conv2d(input=pool3,
                                num_filters=128,
                                filter_size=2,
                                param_attr=ParamAttr(initializer=Xavier(),
                                                     regularizer=L2DecayRegularizer(0.0005)),
                                name='conv4')
    conv4_prelu = fluid.layers.prelu(x=conv4, mode='all', name='conv4_prelu')

    # 把图像特征进行展开
    fc_flatten = fluid.layers.flatten(conv4_prelu)

    # 第一层全连接层
    fc1 = fluid.layers.fc(input=fc_flatten, size=256, name='fc1')

    # 是否人脸的分类输出层
    cls_prob = fluid.layers.fc(input=fc1, size=2, act='softmax', name='cls_fc')
    # 是否人脸分类输出交叉熵损失函数
    cls_loss = cls_ohem(cls_prob=cls_prob, label=label)

    # 人脸box的输出层
    bbox_pred = fluid.layers.fc(input=fc1, size=4, act=None, name='bbox_fc')
    # 人脸box的平方差损失函数
    bbox_loss = bbox_ohem(bbox_pred=bbox_pred, bbox_target=bbox_target, label=label)

    # 人脸5个关键点的输出层
    landmark_pred = fluid.layers.fc(input=fc1, size=10, act=None, name='landmark_fc')
    # 人脸关键点的平方差损失函数
    landmark_loss = landmark_ohem(landmark_pred=landmark_pred, landmark_target=landmark_target, label=label)

    # 准确率函数
    accuracy = cal_accuracy(cls_prob=cls_prob, label=label)
    return image, label, bbox_target, landmark_target, cls_loss, bbox_loss, landmark_loss, accuracy, cls_prob, bbox_pred, landmark_pred


# 用于自定义op创建张量
def create_tmp_var(name, dtype, shape):
    return fluid.default_main_program().current_block().create_var(name=name, dtype=dtype, shape=shape)


# 是否有人脸交叉熵损失函数
def cls_ohem(cls_prob, label):
    # 只把pos的label设定为1,其余都为0
    def my_where1(zeros, label):
        label_filter_invalid = np.where(np.less(label, 0), zeros, label)
        return label_filter_invalid

    zeros = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='int64', value=0)
    label_filter_invalid = create_tmp_var(name='label_filter_invalid', dtype='int64', shape=label.shape)
    label_filter_invalid = fluid.layers.py_func(func=my_where1, x=[zeros, label], out=label_filter_invalid)

    loss = fluid.layers.cross_entropy(input=cls_prob, label=label_filter_invalid)
    # 只取70%的数据
    loss = fluid.layers.squeeze(input=loss, axes=[])
    loss, _ = fluid.layers.topk(input=loss, k=268)
    # loss = fluid.layers.reduce_mean(loss)
    return loss


# 人脸box的平方差损失函数
def bbox_ohem(bbox_pred, bbox_target, label):
    # 保留pos和part的数据
    def my_where3(label, ones_index, zeros_index):
        valid_inds = np.where(np.equal(np.abs(label), 1), ones_index, zeros_index)
        return valid_inds

    zeros = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='float32', value=0)
    ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='float32', value=1)
    valid_inds = create_tmp_var(name='valid_inds_bbox_ohem', dtype='float32', shape=label.shape)
    valid_inds = fluid.layers.py_func(func=my_where3, x=[label, ones, zeros], out=valid_inds)

    square_error = fluid.layers.square_error_cost(input=bbox_pred, label=bbox_target)
    square_error = square_error * valid_inds
    square_error = fluid.layers.reduce_mean(square_error)
    return square_error


# 关键点的平方差损失函数
def landmark_ohem(landmark_pred, landmark_target, label):
    # 只保留landmark数据
    def my_where4(label, ones, zeros):
        valid_inds = np.where(np.equal(label, -2), ones, zeros)
        return valid_inds

    ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='float32', value=1)
    zeros = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='float32', value=0)
    valid_inds = create_tmp_var(name='valid_inds_landmark_ohem', dtype='float32', shape=label.shape)
    valid_inds = fluid.layers.py_func(func=my_where4, x=[label, ones, zeros], out=valid_inds)

    square_error = fluid.layers.square_error_cost(input=landmark_pred, label=landmark_target)
    square_error = square_error * valid_inds
    square_error = fluid.layers.reduce_mean(square_error)
    return square_error


# 计算分类准确率
def cal_accuracy(cls_prob, label):
    # 保留label>=0的数据，即pos和neg的数据
    def my_where1(zeros, ones, label):
        label_filter_invalid = np.where(np.greater_equal(label, 0), ones, zeros)
        return label_filter_invalid

    zeros = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='int64', value=0)
    ones = fluid.layers.fill_constant_batch_size_like(input=label, shape=label.shape, dtype='int64', value=1)
    picked = create_tmp_var(name='picked', dtype='int64', shape=label.shape)
    picked = fluid.layers.py_func(func=my_where1, x=[zeros, ones, label], out=picked)

    accuracy_op = fluid.layers.accuracy(input=cls_prob, label=picked)
    return accuracy_op


# 训练的优化方法
def optimize(loss, data_num, batch_size):
    '''参数优化'''
    lr_factor = 0.1
    base_lr = 0.001
    LR_EPOCH = [6, 14, 20]
    boundaries = [int(epoch * data_num / batch_size) for epoch in LR_EPOCH]
    lr_values = [base_lr * (lr_factor ** x) for x in range(0, len(LR_EPOCH) + 1)]
    lr_op = fluid.layers.piecewise_decay(boundaries=boundaries, values=lr_values)
    optimizer = fluid.optimizer.Momentum(learning_rate=lr_op, momentum=0.9)
    train_op, _ = optimizer.minimize(loss)
    return train_op, lr_op
