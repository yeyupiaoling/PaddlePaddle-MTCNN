import paddle.nn as nn
import paddle


class PNet(nn.Layer):
    def __init__(self):
        super(PNet, self).__init__(name_scope='PNet')
        weight_attr = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0.0005))
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=10, kernel_size=3, padding='valid', weight_attr=weight_attr)
        self.pool1 = nn.MaxPool2D(kernel_size=2, stride=2, padding='same')
        self.conv2 = nn.Conv2D(in_channels=10, out_channels=16, kernel_size=3, padding='valid', weight_attr=weight_attr)
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding='valid', weight_attr=weight_attr)
        self.conv4_1 = nn.Conv2D(in_channels=32, out_channels=2, kernel_size=1, padding='valid', weight_attr=weight_attr)
        self.conv4_2 = nn.Conv2D(in_channels=32, out_channels=4, kernel_size=1, padding='valid', weight_attr=weight_attr)
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax(axis=1)

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.conv3(x))
        # 分类是否人脸的卷积输出层
        class_out = self.conv4_1(x)
        class_out = paddle.squeeze(class_out, axis=[2, 3])
        class_out = self.softmax(class_out)
        # 人脸box的回归卷积输出层
        bbox_out = self.conv4_2(x)
        bbox_out = paddle.squeeze(bbox_out, axis=[2, 3])
        return class_out, bbox_out
