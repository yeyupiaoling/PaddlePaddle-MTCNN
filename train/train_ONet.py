import os
import shutil
from model import O_Net, optimize
import myreader
import paddle.fluid as fluid
import config as cfg

radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 1.0
batch_size = 384

# 获取O网络
image, label, bbox_target, landmark_target, cls_loss, bbox_loss, landmark_loss, accuracy, cls_prob, bbox_pred, landmark_pred = O_Net()

# 获取训练的损失函数
total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + radio_landmark_loss * landmark_loss
avg_total_loss = fluid.layers.mean(total_loss)

# 计算一共多少组数据
label_file = '../data/48/all_data.label'
f = open(label_file, 'r')
num = len(f.readlines())

# 定义优化方法
_, learning_rate = optimize(avg_total_loss, num, batch_size)

# 获取自定义数据
train_reader = myreader.train_reader('../data/48/all_data', label_file, batch_size=cfg.BATCH_SIZE)

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 设置输出的结果
fetch_list = [avg_total_loss, accuracy, learning_rate, cls_loss, bbox_loss, landmark_loss]

# 训练
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, acc, lr, cls_loss1, bbox_loss1, landmark_loss1 = exe.run(program=fluid.default_main_program(),
                                                                             feed={image.name: data[0],
                                                                                   label.name: data[1],
                                                                                   bbox_target.name: data[2],
                                                                                   landmark_target.name: data[3]},
                                                                             fetch_list=fetch_list)

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, cls_loss:%0.5f, box_loss:%0.5f, landmark_loss : %0.5f, Accuracy：'
                  '%0.5f, Learning rate:%0.7f' % (pass_id, batch_id, train_cost[0], cls_loss1[0],
                                                  bbox_loss1[0], landmark_loss1[0], acc[0], lr[0]))
    # 保存预测模型
    save_path = '../infer_model/ONet/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(dirname=save_path,
                                  feeded_var_names=[image.name],
                                  target_vars=[cls_prob, bbox_pred, landmark_pred],
                                  executor=exe)
