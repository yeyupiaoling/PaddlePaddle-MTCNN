import paddle.nn as nn
import paddle


class PNet(nn.Layer):
    def __init__(self):
        super(PNet, self).__init__(name_scope='PNet')
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=10, kernel_size=3, padding="VALID")
        self.pool1 = nn.Pool2D(pool_size=2, pool_stride=2, pool_type='max')
        self.conv2 = nn.Conv2D(in_channels=10, out_channels=16, kernel_size=3, padding="VALID")
        self.conv3 = nn.Conv2D(in_channels=16, out_channels=32, kernel_size=3, padding="VALID")
        self.conv4_1 = nn.Conv2D(in_channels=32, out_channels=2, kernel_size=1, padding="VALID")
        self.conv4_2 = nn.Conv2D(in_channels=32, out_channels=4, kernel_size=1, padding="VALID")
        self.conv4_3 = nn.Conv2D(in_channels=32, out_channels=10, kernel_size=1, padding="VALID")
        self.prelu = nn.PReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu(self.conv2(x))
        x = self.prelu(self.conv3(x))
        # 分类是否人脸的卷积输出层
        class_out = self.conv4_1(x)
        class_out = paddle.squeeze(class_out)
        class_out = self.softmax(class_out)
        # 人脸box的回归卷积输出层
        bbox_out = self.conv4_2(x)
        bbox_out = paddle.squeeze(bbox_out)
        # 5个关键点的回归卷积输出层
        landmark_out = self.conv4_3(x)
        landmark_out = paddle.squeeze(landmark_out)
        return class_out, bbox_out, landmark_out
