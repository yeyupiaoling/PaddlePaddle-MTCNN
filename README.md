# PaddlePaddle-MTCNN
基于PaddlePaddle复现的MTCNN人脸检测模型


## train目录
 - `train/config.py` 训练和模型配置参数
 - `train/data_format_converter.py` 把大量的图片合并成一个文件
 - `train/generate_ONet_data.py` 生成ONet训练的数据
 - `train/generate_PNet_data.py` 生成PNet训练的数据
 - `train/generate_RNet_data.py` 生成RNet训练的数据
 - `train/infer.py` 预测图像程序，检测图片上人脸的位置和关键的位置
 - `train/model.py` 三个模型的定义、损失函数、优化方法的定义
 - `train/myreader.py` 读取已经合并数目的reader
 - `train/train_ONet.py` 训练ONet网络模型
 - `train/train_PNet.py` 训练PNet网络模型
 - `train/train_RNet.py` 训练RNet网络模型
 - `train/utils.py` 所用到的工具类
 
 
## 其他目录
 - `data` 存放训练数据
 - `infer_model` 存放训练保存的预测模型

## 数据集下载
 - [WIDER Face](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/) 下载训练数据WIDER Face Training Images，解压的WIDER_train文件夹放置到dataset下。并下载 [Face annotations](http://mmlab.ie.cuhk.edu.hk/projects/WIDERFace/support/bbx_annotation/wider_face_split.zip) ，解压把里面的 wider_face_train_bbx_gt.txt 文件放在dataset目录下，
 - [Deep Convolutional Network Cascade for Facial Point Detection](http://mmlab.ie.cuhk.edu.hk/archive/CNN_FacePoint.htm) 。下载 Training set 并解压，将里面的 lfw_5590 和 net_7876 文件夹放置到dataset下
 - 解压数据集之后，`dataset`目录下应该有文件夹`lfw_5590`，`net_7876`，`WIDER_train`，有标注文件`testImageList.txt`，`trainImageList.txt`，`wider_face_train.txt`，`wider_face_train_bbx_gt.txt`

 
## 第一步 训练PNet模型
 - `python3 train/generate_PNet_data.py` 首先需要生成PNet模型训练所需要的数据
 - `python3 train/train_PNet.py` 开始训练PNet模型

## 第二步 训练RNet模型
 - `python3 train/generate_RNet_data.py` 使用上一步训练好的PNet模型生成RNet训练所需的模型
 - `python3 train/train_RNet.py` 开始训练RNet模型


## 第三步 训练ONet模型
 - `python3 train/generate_ONet_data.py` 使用上两部步训练好的PNet模型和RNet模型生成RNet训练所需的模型
 - `python3 train/train_ONet.py` 开始训练ONet模型


## 第四部 预测图片
 - `python3 train/infer.py` 使用上面训练得到的三个模型执行预测，识别图片中人脸box和关键点


## 参考资料
1. https://github.com/AITTSMD/MTCNN-Tensorflow





