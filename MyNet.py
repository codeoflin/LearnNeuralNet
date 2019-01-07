import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

# region 常量
# 求导数的时候用的极小值
H = 1e-4
# endregion

# region 激活函数
def Sigmoid(x):
    return 1/(1+np.exp(-x))

# 恒等函数
def indentity_func(x):
    return x
# endregion 激活函数


# region 损失函数 参数:神经网络输出,正确数据
# 平方差
def Mean_Square_Error(y, t):
    return 0.5*np.sum((y-t)**2)

# 交叉熵
def Cross_Entropy_Error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0-np.sum(t * np.log(y + 1e-7)) / batch_size
# endregion

# region 偏导数 算法
def _numerical_gradient_no_batch(f, x):
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)

    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # f(x+h)

        x[idx] = tmp_val - h
        fxh2 = f(x)  # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)

        x[idx] = tmp_val  # 还原值
    return grad


def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)

        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)

        return grad
# endregion

class Net:
    #region 网络
    # 这里初始化一个W,B网络
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        self.network = {}
        self.network["W1"] = weight_init_std / \
            np.random.rand(input_size, hidden_size)
        self.network["B1"] = np.zeros(hidden_size)

        self.network["W2"] = weight_init_std / \
            np.random.rand(hidden_size, output_size)
        self.network["B2"] = np.zeros(output_size)

    # 前向运算
    def predict(self, x):
        w1, w2 = self.network["W1"], self.network["W2"]
        b1, b2 = self.network['B1'], self.network['B2']

        # 输入层
        z1 = Sigmoid(np.dot(x, w1)+b1)

        # 输出层
        y = indentity_func(np.dot(z1, w2)+b2)
        return y
    # endregion

    #region 损失函数 x:输入数据, t:监督数据
    def loss(self, x, t):
        y = self.predict(x)
        return Cross_Entropy_Error(y, t)
    #endregion

    #region 正确率
    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)
        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy
    # endregion

    # region 梯度下降
    # 梯度生成算法
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        def loss_W(W): return self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.network['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.network['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.network['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.network['B2'])
        return grads

    def gradient_desent(self, x, t, lr=1, step_num=100):
        for i in range(step_num):
            grad = self.numerical_gradient(x, t)
            self.network['W1'] -= grad['W1']*lr
            self.network['B1'] -= grad['B1']*lr
            self.network['W2'] -= grad['W2']*lr
            self.network['B2'] -= grad['B2']*lr
        pass

    # endregion

def _change_one_hot_label(X):
    T = np.zeros((X.size, 2))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
        
    return T

# 测试网络
def nettest():
    # 生成0~30之间的数字
    x = np.arange(0,30,1).reshape(30,1)

    # 计算它的 单/双 以onehot格式输出
    t =_change_one_hot_label(np.remainder(x,2))

    # 准备神经网络
    net = Net(1, 4, 2)

    # 训练前输出效果
    print(net.accuracy(x,t))

    # 训练
    net.gradient_desent(x,t,0.1,1000)

    # 训练后输出效果
    print(net.accuracy(x,t))

    # 分类
    #y = net.predict(x)
    #print(y)

if __name__ == "__main__":
    nettest()
    pass
