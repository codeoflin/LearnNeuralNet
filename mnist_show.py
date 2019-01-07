# coding: utf-8
import sys
import os
# sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
from mnist import load_mnist
from PIL import Image
import pickle

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

def test1():
    (x_train, t_train), (x_test, t_test) = load_mnist(
        flatten=True, normalize=False)
    img = x_train[0]
    label = t_train[0]
    print(label)  # 5
    print(img.shape)  # (784,)
    img = img.reshape(28, 28)  # 把图像的形状变为原来的尺寸
    print(img.shape)  # (28, 28)
    img_show(img)

# 感知机
def Perceptron(x):
    return np.array(x > 0, dtype=np.int)

# S
def Sigmoid(x):
    return 1/(1+np.exp(-x))

# Relu
def Relu(x):
    return np.maximum(0, x)

# Softmax
def Softmax(a):
    c=np.max(a)
    exp_a = np.exp(a-c)
    y = exp_a / np.sum(exp_a)
    return y

def LoadData():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test,t_test

def init_network():
    with open("sample_weight.pkl",'rb') as f:
        network=pickle.load(f)
    return network

def forward(network,x):
    w1,w2,w3=network['W1'],network['W2'],network['W3']
    b1,b2,b3=network['b1'],network['b2'],network['b3']
    # 隐含层1
    a1=np.dot(x,w1)+b1
    z1=Sigmoid(a1)

    # 隐含层2
    a2=np.dot(z1,w2)+b2
    z2=Sigmoid(a2)
    
    # 输出层
    a3=np.dot(z2,w3)+b3
    y=Softmax(a3)

    return y

def test2():
    x,t=LoadData()
    network=init_network()

    #y=forward(network,x)
    #print(y[0])

    accuracy_cnt = 0
    for i in range(len(x)):
        y = forward(network, x[i])
        p = np.argmax(y) # 获取概率最高的元素的索引
        if p == t[i]:
            accuracy_cnt += 1
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
    #forword()

def test3():
    x,t=LoadData()
    network=init_network()

    batch_size=1000
    accuracy_cnt = 0
    for i in range(0,len(x),batch_size):
        x_batch=x[i:i+batch_size]
        y_batch = forward(network, x_batch)
        p = np.argmax(y_batch, axis=1)
        accuracy_cnt += np.sum(p == t[i:i+batch_size])
    print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
if (__name__=='__main__'):
    test3()