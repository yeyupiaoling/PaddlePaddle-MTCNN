import paddle.nn as nn


class RNet(nn.Layer):
    def __init__(self):
        super(RNet, self).__init__(name_scope='RNet')
        self.conv1 = nn.Conv2D(in_channels=3, out_channels=28, kernel_size=3, padding="VALID")
        self.pool1 = nn.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
        self.conv2 = nn.Conv2D(in_channels=28, out_channels=48, kernel_size=3, padding="VALID")
        self.pool2 = nn.Pool2D(pool_size=3, pool_stride=2, pool_type='max')
        self.conv3 = nn.Conv2D(in_channels=48, out_channels=64, kernel_size=2, padding="VALID")
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(in_features=256, out_features=128)
        self.class_fc = nn.Linear(in_features=128, out_features=2)
        self.softmax = nn.Softmax()
        self.bbox_fc = nn.Linear(in_features=128, out_features=4)
        self.landmark_fc = nn.Linear(in_features=128, out_features=10)
        self.prelu = nn.PReLU()

    def forward(self, x):
        x = self.prelu(self.conv1(x))
        x = self.pool1(x)
        x = self.prelu(self.conv2(x))
        x = self.pool2(x)
        x = self.prelu(self.conv3(x))
        x = self.flatten(x)
        x = self.fc(x)
        # 分类是否人脸的卷积输出层
        class_out = self.class_fc(x)
        class_out = self.softmax(class_out)
        # 人脸box的回归卷积输出层
        bbox_out = self.bbox_fc(x)
        # 5个关键点的回归卷积输出层
        landmark_out = self.landmark_fc(x)
        return class_out, bbox_out, landmark_out
