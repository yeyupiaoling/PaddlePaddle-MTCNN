import os
import shutil
from model import R_Net, optimize
import paddle as paddle
import reader
import paddle.fluid as fluid

radio_cls_loss = 1.0
radio_bbox_loss = 0.5
radio_landmark_loss = 0.5
batch_size = 32

# 获取R网络
image, label, bbox_target, landmark_target, cls_loss, bbox_loss, landmark_loss, accuracy, cls_prob, bbox_pred, landmark_pred = R_Net()

# 获取训练损失函数
total_loss = radio_cls_loss * cls_loss + radio_bbox_loss * bbox_loss + radio_landmark_loss * landmark_loss
avg_total_loss = fluid.layers.mean(total_loss)

# 计算一共多少组数据
label_file = '../data/24/all_24_data.txt'
f = open(label_file, 'r')
num = len(f.readlines())

# 定义优化方法
_, learning_rate = optimize(avg_total_loss, num, batch_size)

# 获取自定义数据
train_reader = paddle.batch(reader=reader.train_reader('../data/24/all_24_data.txt'), batch_size=batch_size)

# 定义一个使用GPU的执行器
place = fluid.CUDAPlace(0)
# place = fluid.CPUPlace()
exe = fluid.Executor(place)
# 进行参数初始化
exe.run(fluid.default_startup_program())

# 定义输入数据维度
feeder = fluid.DataFeeder(place=place, feed_list=[image, label, bbox_target, landmark_target])

# 训练
for pass_id in range(100):
    # 进行训练
    for batch_id, data in enumerate(train_reader()):
        train_cost, acc, lr = exe.run(program=fluid.default_main_program(),
                                      feed=feeder.feed(data),
                                      fetch_list=[avg_total_loss, accuracy, learning_rate])

        # 每100个batch打印一次信息
        if batch_id % 100 == 0:
            print('Pass:%d, Batch:%d, Cost:%0.5f, Accuracy：%0.5f, Learning rate:%0.7f' % (
            pass_id, batch_id, train_cost[0], acc[0], lr[0]))

    # 保存预测模型
    save_path = 'infer_model/RNet/'
    # 删除旧的模型文件
    shutil.rmtree(save_path, ignore_errors=True)
    # 创建保持模型文件目录
    os.makedirs(save_path)
    # 保存预测模型
    fluid.io.save_inference_model(dirname=save_path,
                                  feeded_var_names=[image.name],
                                  target_vars=[cls_prob, bbox_pred, landmark_pred],
                                  executor=exe)
