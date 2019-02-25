import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import time
import datetime

# region 常量
# 求导数的时候用的极小值
H = 1e-4
# endregion

# region 激活函数
def Sigmoid(x):
    return 1/(1+np.exp(-x))

def Sigmoid_grad(x):
    return (1.0 - Sigmoid(x)) * Sigmoid(x)

# Softmax
def Softmax(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    y = exp_a / np.sum(exp_a)
    return y

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
        np.random.seed(int(time.time()))
        self.network["W1"] = weight_init_std / np.random.randn(input_size, hidden_size)
        self.network["B1"] = np.zeros(hidden_size)

        self.network["W2"] = weight_init_std / np.random.randn(hidden_size, output_size)
        self.network["B2"] = np.zeros(output_size)

    # 前向运算
    def predict(self, x):
        w1, w2 = self.network["W1"], self.network["W2"]
        b1, b2 = self.network['B1'], self.network['B2']

        # 输入层
        z1 = Sigmoid(np.dot(x, w1)+b1)

        # 输出层
        y = Softmax(np.dot(z1, w2)+b2)
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

    #region 梯度下降
    # 梯度生成算法
    # x:输入数据, t:监督数据
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.network['W1'])
        grads['B1'] = numerical_gradient(loss_W, self.network['B1'])
        grads['W2'] = numerical_gradient(loss_W, self.network['W2'])
        grads['B2'] = numerical_gradient(loss_W, self.network['B2'])
        return grads

    # 反向传播
    def gradient(self, x, t):
        W1, W2 = self.network['W1'], self.network['W2']
        b1, b2 = self.network['B1'], self.network['B2']
        grads = {}
        
        batch_num = x.shape[0]
        
        # forward
        a1 = np.dot(x, W1) + b1
        z1 = Sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = Softmax(a2)
        
        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['B2'] = np.sum(dy, axis=0)
        
        da1 = np.dot(dy, W2.T)
        dz1 = Sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['B1'] = np.sum(dz1, axis=0)
        return grads
    #endregion
    
    # 训练
    def gradient_desent(self, x, t, lr=1, step_num=100):
        for i in range(step_num):
            grad = self.numerical_gradient(x, t)
            self.network['W1'] -= grad['W1']*lr
            self.network['B1'] -= grad['B1']*lr
            self.network['W2'] -= grad['W2']*lr
            self.network['B2'] -= grad['B2']*lr
        pass
    
def _change_one_hot_label(X,len):
    T = np.zeros((X.size, len))
    for idx, row in enumerate(T):
        row[X[idx]] = 1
    return T

# 测试网络
def nettest():
    # 生成0~1之间的数字
    x = np.arange(0,1,0.01)
    x = x.reshape(x.size,1)

    # 计算它的 单/双 以onehot格式输出
    # t =_change_one_hot_label((np.remainder(x,2)==0).astype(np.int),2)

    # 判断是否大于0.05
    # t =_change_one_hot_label((x>0.05).astype(np.int),2)

    # 把数字分为10个区域
    t =_change_one_hot_label((x/0.1).astype(np.int),10)

    # 写一个函数 尝试拟合他
    # t=x

    # 准备神经网络
    net = Net(1, 8, 10)

    # 训练前输出效果
    print(net.accuracy(x,t)) #0.0

    # 训练
    net.gradient_desent(x,t,0.1,10000)

    # 训练后输出效果
    print(net.accuracy(x,t)) #0.0

    # 分类
    #y = net.predict(x)
    #print(y)

if __name__ == "__main__":
    nettest()
    pass
