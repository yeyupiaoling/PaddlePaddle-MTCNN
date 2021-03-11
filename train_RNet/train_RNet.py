import os
import sys
from datetime import datetime

import paddle
from paddle.io import DataLoader

sys.path.append("../")

from models.Loss import ClassLoss, BBoxLoss, LandmarkLoss, accuracy
from models.RNet import RNet
from utils.data import CustomDataset

# 设置损失值的比例
radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5

# 训练参数值
data_path = '../dataset/24/all_data'
batch_size = 384
learning_rate = 1e-3
epoch_num = 30
model_path = '../infer_models'

# 获取数据
train_dataset = CustomDataset(data_path)
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

# 获取P模型
model = RNet()

# 设置优化方法
scheduler = paddle.optimizer.lr.PiecewiseDecay(boundaries=[6, 14, 20], values=[0.001, 0.0001, 0.00001, 0.000001],
                                               verbose=True)
optimizer = paddle.optimizer.Momentum(parameters=model.parameters(), learning_rate=scheduler)

# 获取损失函数
class_loss = ClassLoss()
bbox_loss = BBoxLoss()
landmark_loss = LandmarkLoss()

# 开始训练
for epoch in range(epoch_num):
    for batch_id, (img, label, bbox, landmark) in enumerate(train_loader()):
        class_out, bbox_out, landmark_out = model(img)
        cls_loss = class_loss(class_out, label)
        box_loss = bbox_loss(bbox_out, bbox, label)
        landmarks_loss = landmark_loss(landmark_out, landmark, label)
        total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * box_loss + radio_landmark_loss * landmarks_loss
        total_loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if batch_id % 100 == 0:
            acc = accuracy(class_out, label)
            print('[%s] Train epoch %d, batch %d, loss: %f, accuracy：%f' % (
                datetime.now(), epoch, batch_id, total_loss, acc))
    scheduler.step()

    # 保存模型
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    paddle.save(model.state_dict(), os.path.join(model_path, 'RNet.pdparams'))
