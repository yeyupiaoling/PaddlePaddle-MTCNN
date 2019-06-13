# PaddlePaddle-MTCNN
基于PaddlePaddle复现的MTCNN人脸检测模型


# train目录
 - `train/config.py` 训练和模型配置参数
 - `train/data_format_converter.py` 把大量的图片合并成一个文件
 - `train/generate_ONet_data.py` 生成ONet训练的数据
 - `train/generate_PNet_data.py` 生成PNet训练的数据
 - `train/generate_RNet_data.py` 生成RNet训练的数据
 - `train/model.py` 三个模型的定义、损失函数、优化方法的定义
 - `train/reader.py` 训练数据的读取的reader
 - `train/myreader.py` 读取已经合并数目的reader
 - `train/train_ONet.py` 训练ONet网络模型
 - `train/train_PNet.py` 训练PNet网络模型
 - `train/train_RNet.py` 训练RNet网络模型
 - `train/utils.py` 所用到的工具类
 
 
# 其他目录
 - `data` 存放训练数据
 - `infer_model` 存放训练保存的预测模型
 
# 训练PNet模型
 - `python3 train/generate_PNet_data.py` 首先需要生成PNet模型训练所需要的数据
 - `python3 train/train_PNet.py` 开始训练PNet模型