import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
from mpl_toolkits.mplot3d import Axes3D

# 与门 权重偏置
AND_W = np.array([0.5, 0.5])
AND_B = -0.7

# 与非门 权重偏置
NAND_W = np.array([-0.5, -0.5])
NAND_B = 0.7

# 或门 权重偏置
OR_W = np.array([0.5, 0.5])
OR_B = -0.3

# 感知机
def Perceptron(x):
    return np.array(x > 0, dtype=np.int)

# sigmoid
def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Relu
def Relu(x):
    return np.maximum(0, x)

# 这个激活函数在"分类"中,比较常用
def Softmax(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    y = exp_a / np.sum(exp_a)
    return y

# 异或门
def XOR(x1, x2):
    local_x1 = Perceptron(np.sum(np.array([x1, x2])*NAND_W)+NAND_B)
    local_x2 = Perceptron(np.sum(np.array([x1, x2])*OR_W)+OR_B)
    return Sigmoid(np.sum(np.array([local_x1, local_x2])*AND_W)+AND_B)

def test1():
    print(XOR(0, 0))
    print(XOR(1, 0))
    print(XOR(0, 1))
    print(XOR(1, 1))

def test2():
    x = np.arange(-10, 10, 0.1)
    y1 = Perceptron(x)
    y2 = Sigmoid(x)
    plt.plot(x, x,linestyle='--',label="x")
    plt.plot(x, y1,label="Perceptron")
    plt.plot(x, y2,label="Sigmoid")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(-0.01, 1.01)  # 指定y轴的范围
    plt.legend()
    plt.show()

#第三章
def identity_function(x):
    return x


def test3():
    # 输入层
    X = np.array([1.0, 0.5])#输入层
    W1 = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])#输入层权重
    B1 = np.array([0.1, 0.2, 0.3])#输入层偏置
    A1 = np.dot(X, W1) + B1
    Z1 = Sigmoid(A1)
    #print(A1) # [0.3, 0.7, 1.1]
    #print(Z1) # [0.57444252, 0.66818777, 0.75026011]

    # 隐含层
    W2 = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    B2 = np.array([0.1, 0.2])
    A2 = np.dot(Z1, W2) + B2
    Z2 = Sigmoid(A2)

    # 输出层
    W3 = np.array([[0.1, 0.3], [0.2, 0.4]])
    B3 = np.array([0.1, 0.2])
    A3 = np.dot(Z2, W3) + B3
    Y = identity_function(A3) # 或者Y = A3
    print(Y)

# 把test3规整下:
# 网络权重和偏置
def init_network():
    network = {}
    network['W1'] = np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
    network['b1'] = np.array([0.1, 0.2, 0.3])
    network['W2'] = np.array([[0.1, 0.4], [0.2, 0.5], [0.3, 0.6]])
    network['b2'] = np.array([0.1, 0.2])
    network['W3'] = np.array([[0.1, 0.3], [0.2, 0.4]])
    network['b3'] = np.array([0.1, 0.2])
    return network

# 前向运算
def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = Sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = Sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)
    return y

# 测试数据
def test4():
    network = init_network()
    x = np.array([1.0, 0.5])
    y = forward(network, x)
    print(y) # [ 0.31682708 0.69627909]

def test5():
    a=np.array([0.3,2.9,4.0])
    y=Softmax(a)
    print(y)


def APerceptron(w,b,x):
    y=np.sum(x*w)+b

    # 激活函数
    if(y>=0):
        return 1
    else:
        return 0


# 第一层
XOR_W1_1 = np.array([0.5, 0.5])
XOR_B1_1 = -0.7

XOR_W1_2= np.array([-0.5, -0.5])
XOR_B1_2 = 0.3

# 第二层
XOR_W2 = np.array([-0.5,-0.5])
XOR_B2 = 0.3

# 这是一个两层网络,输入层(2个神经元) + 输出层(1个神经元)
def TwoLayerNet(x):
    # 输入层第1神经元
    a1=APerceptron(XOR_W1_1,XOR_B1_1,x )
    # 输入层第2神经元
    a2=APerceptron(XOR_W1_2,XOR_B1_2,x )
    y=APerceptron(XOR_W2,XOR_B2,np.array([a1,a2]) )
    return y

def test0():
    print (TwoLayerNet(np.array([0,0]) ))
    print (TwoLayerNet(np.array([1,0]) ))
    print (TwoLayerNet(np.array([1,1]) ))
    print (TwoLayerNet(np.array([0,1]) ))

if __name__ == "__main__":
    # x=np.array([-1.0,1.0,2.0])
    # x=Sigmoid(x)
    # print(x)
    test2()