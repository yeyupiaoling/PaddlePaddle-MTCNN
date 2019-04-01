# PaddlePaddle-MTCNN
基于PaddlePaddle复现的MTCNN人脸检测模型


# train目录
 - `train/generate_ONet_data.py` 生成ONet训练的数据
 - `train/generate_PNet_data.py` 生成PNet训练的数据
 - `train/generate_RNet_data.py` 生成RNet训练的数据
 - `train/model.py` 三个模型的定义、损失函数、优化方法的定义
 - `train/reader.py` 训练数据的读取的reader
 - `train/train_ONet.py` 训练ONet网络模型
 - `train/train_PNet.py` 训练PNet网络模型
 - `train/train_RNet.py` 训练RNet网络模型
 - `train/utils.py` 所用到的工具类
 
 
# 其他目录
 - `data` 存放主要的训练数据
 - `data2` 存放的是便于上传到Github给其他人调试的少量数据
 - `infer_model` 存放训练保存的预测模型