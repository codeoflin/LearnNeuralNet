import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

#region 常量
# 求导数的时候用的极小值
H=1e-4
#endregion

class Net:
#region 网络
    #region 激活函数
    @staticmethod
    def Sigmoid(x):
        return 1/(1+np.exp(-x))

    # 恒等函数
    
    @staticmethod
    def indentity_func(x):
        return x
    #endregion 激活函数

    # 这里初始化一个W,B网络
    def __init__(self):
        self.network = {}
        self.network["W1"] = np.random.rand(1)
        self.network["B1"] = np.random.rand(1)

        self.network["W2"] = np.random.rand(1,2)
        self.network["B2"] = np.random.rand(1,2)

    # 前向运算
    def forward(self, x):
        w1, w2 = self.network["W1"], self.network["W2"]
        b1, b2 = self.network['B1'], self.network['B2']

        # 输入层
        z1 = Net.Sigmoid(np.dot(x, w1)+b1)

        # 输出层
        y = Net.indentity_func(np.dot(z1, w2)+b2)
        return y
    #endregion

    #region 损失函数 参数:神经网络输出,正确数据
    # 平方差
    @staticmethod
    def Mean_Square_Error(y,t):
        return 0.5*np.sum((y-t)**2)
        
    # 交叉熵
    @staticmethod
    def Cross_Entropy_Error(y, t):
        if y.ndim == 1:
            t = t.reshape(1, t.size)
            y = y.reshape(1, y.size)
        batch_size = y.shape[0]
        return 0-np.sum(t * np.log(y + 1e-7)) / batch_size

    # x:输入数据, t:监督数据
    def loss(self,x, t):
        y = self.forward(x)
        return Net.Cross_Entropy_Error(y, t)
    #endregion

    #region 梯度运算

    #endregion

# 测试网络
def nettest():
    net=Net()
    x=np.array([0])
    y=net.forward(x)
    print(y)

if __name__ == "__main__":
    nettest()
    pass