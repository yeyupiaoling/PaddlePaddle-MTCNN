import paddle.fluid as fluid
from paddle.fluid.param_attr import ParamAttr
from paddle.fluid.initializer import Xavier
from paddle.fluid.regularizer import L2DecayRegularizer
import numpy as np
import config as cfg


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
    conv4_1 = fluid.layers.transpose(conv4_1, [0, 2, 3, 1])
    conv4_1 = fluid.layers.squeeze(input=conv4_1, axes=[])
    conv4_1_softmax = fluid.layers.softmax(input=conv4_1)

    # 人脸box的回归卷积输出层
    conv4_2 = fluid.layers.conv2d(input=conv3_prelu,
                                  num_filters=4,
                                  filter_size=1,
                                  param_attr=ParamAttr(initializer=Xavier(),
                                                       regularizer=L2DecayRegularizer(0.0005)),
                                  name='conv4_2')
    conv4_2 = fluid.layers.transpose(conv4_2, [0, 2, 3, 1])

    # 5个关键点的回归卷积输出层
    conv4_3 = fluid.layers.conv2d(input=conv3_prelu,
                                  num_filters=10,
                                  filter_size=1,
                                  param_attr=ParamAttr(initializer=Xavier(),
                                                       regularizer=L2DecayRegularizer(0.0005)),
                                  name='conv4_3')
    conv4_3 = fluid.layers.transpose(conv4_3, [0, 2, 3, 1])

    # 获取是否人脸分类交叉熵损失函数
    cls_prob = fluid.layers.squeeze(input=conv4_1_softmax, axes=[], name='cls_prob')
    cls_loss = cls_ohem(cls_prob=cls_prob, label=label)

    # 获取人脸box回归平方差损失函数
    bbox_pred = fluid.layers.squeeze(input=conv4_2, axes=[], name='bbox_pred')
    bbox_loss = bbox_ohem(bbox_pred=bbox_pred, bbox_target=bbox_target, label=label)

    # 获取人脸5个关键点回归平方差损失函数
    landmark_pred = fluid.layers.squeeze(input=conv4_3, axes=[], name='landmark_pred')
    landmark_loss = landmark_ohem(landmark_pred=landmark_pred, landmark_target=landmark_target, label=label)
    # 准确率函数
    accuracy = cal_accuracy(cls_prob=cls_prob, label=label)
    return image, label, bbox_target, landmark_target, cls_loss, bbox_loss, landmark_loss, accuracy, cls_prob, bbox_pred, landmark_pred


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


# 是否有人脸交叉熵损失函数
def cls_ohem(cls_prob, label):
    # 自定义where op
    def where_op(x, thresh_value, true_val):
        # 阈值变量
        thresh = fluid.layers.fill_constant([1], dtype='int64', value=thresh_value)
        thresh.stop_gradient = True

        # label中小于0的部分，即 part 和 landmark
        if_cond1 = fluid.layers.less_than(x=x, y=thresh)
        # 将label中小于0的部分，赋值为 true_val, 其余的保持不变
        ie = fluid.layers.IfElse(if_cond1)
        with ie.true_block():
            x1 = ie.input(x)
            fill_val = fluid.layers.fill_constant_batch_size_like(x1, shape=[-1, 1], dtype=np.int64, value=true_val)
            ie.output(fill_val)
        with ie.false_block():
            x2 = ie.input(x)
            ie.output(x2)
        res = ie()
        # 返回新的label
        new_label1 = res[0]
        return new_label1, if_cond1

    # 保留neg 0 和pos 1 的数据，将其他的设置为-100，在计算交叉熵时忽略掉，
    new_label, if_cond = where_op(label, 0, -100)
    new_label.stop_gradient = True
    # bool to float
    cast_if_cond = fluid.layers.cast(if_cond, np.float32)
    # 求保留之后的数量 * 0.7
    keep_num = (cfg.BATCH_SIZE - fluid.layers.reduce_sum(cast_if_cond)) * cfg.KEEP_RATIO
    # 转换为整数
    keep_num = fluid.layers.cast(keep_num, np.int32)
    keep_num.stop_gradient = True
    # 计算损失
    loss = fluid.layers.softmax_with_cross_entropy(cls_prob, new_label, ignore_index=-100)
    # reshape ，求top
    loss1 = fluid.layers.reshape(loss, [1, -1])
    top_loss, top_index = fluid.layers.topk(loss1, k=keep_num)

    top_loss.stop_gradient = True
    top_index.stop_gradient = True

    top_loss = fluid.layers.gather(loss, fluid.layers.reshape(top_index, [-1]))
    return top_loss


# 人脸box的平方差损失函数
def bbox_ohem(bbox_pred, bbox_target, label):
    # 保留pos 1 和part -1 的数据
    thresh = fluid.layers.fill_constant([1], dtype='int64', value=1)
    thresh1 = fluid.layers.fill_constant([1], dtype='int64', value=-1)
    thresh.stop_gradient = True
    thresh1.stop_gradient = True
    if_cond = fluid.layers.logical_or(fluid.layers.equal(label, thresh), fluid.layers.equal(label, thresh1))
    # bool to float
    cast_if_cond = fluid.layers.cast(if_cond, np.float32)
    keep_num = fluid.layers.reduce_sum(cast_if_cond)
    # 转换为整数
    keep_num.stop_gradient = True
    # 求平方差损失
    square_error = fluid.layers.square_error_cost(input=bbox_pred, label=bbox_target)
    square_error = square_error * cast_if_cond

    # avoid divide zero
    square_error = fluid.layers.reduce_sum(square_error) / (fluid.layers.relu(keep_num - 1.0) + 1.0)
    return square_error


# 关键点的平方差损失函数
def landmark_ohem(landmark_pred, landmark_target, label):
    # 只保留landmark数据 -2
    thresh = fluid.layers.fill_constant([1], dtype='int64', value=-2)
    thresh.stop_gradient = True
    if_cond = fluid.layers.equal(x=label, y=thresh)
    # bool to float
    cast_if_cond = fluid.layers.cast(if_cond, np.float32)
    keep_num = fluid.layers.reduce_sum(cast_if_cond)
    keep_num.stop_gradient = True
    square_error = fluid.layers.square_error_cost(input=landmark_pred, label=landmark_target)
    square_error = square_error * cast_if_cond
    square_error = fluid.layers.reduce_sum(square_error) / (fluid.layers.relu(keep_num - 1.0) + 1.0)
    return square_error


# 计算分类准确率
def cal_accuracy(cls_prob, label):
    # 阈值变量
    thresh = fluid.layers.fill_constant([1], dtype='int64', value=1)
    thresh.stop_gradient = True
    # 把label中的1除外，其他的标签都设置为0
    if_cond = fluid.layers.equal(x=label, y=thresh)
    cast_if_cond = fluid.layers.cast(if_cond, np.int64)
    # 求准确率
    accuracy_op = fluid.layers.accuracy(cls_prob, cast_if_cond)
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
