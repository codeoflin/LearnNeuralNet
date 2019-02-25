
import numpy as np
from mnist import load_mnist
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

#np.set_printoptions(suppress=True)

def MeanSquareError(y, t):
    return 0.5*np.sum((y-t)**2)

def CrossEntropyError(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return 0-np.sum(t * np.log(y + 1e-7)) / batch_size

# 试验一下两个损失函数的效果
def test1():
    # 设“2”为正确解
    t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])

    # “2”的概率最高的情况（0.6）
    y = np.array([0.10, 0.05, 0.60, 0.00, 0.05, 0.10, 0.00, 0.10, 0.00, 0.00])

    # print(MeanSquareError(y, t))  # 0.09750000000000003
    # print(CrossEntropyError(y, t))  # 0.510825457099338

    # print(MeanSquareError(y, y))  # 0.0
    # print(CrossEntropyError(y, y))  # 1.2968435295135659

    print(CrossEntropyError(t, t))  # -9.999999505838704e-08

    # 能随机输出数组的函数,以下参数的效果:从0~60000之间随机挑选10个
    print(np.random.choice(60000, 10))

# 抽取部分样品作为神经网络学习数据
def test2():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, one_hot_label=True)
    print(x_train.shape)  # (60000, 784)
    print(t_train.shape)  # (60000, 10)
    train_size = x_train.shape[0]
    batch_size = 10
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

# 一个测试函数
def fun(x):
    return 0.01*x**2+0.1*x

# 求导
def numerical_diff(f,x):
    h=1e-7
    return (f(x+h)-f(x-h))/(h*2)

# 这是根据导数生成线的函数,参数分别是:func,要给func的x, 用于生成线的x数组
def tangent_line(f, xa,x):
    d = numerical_diff(f, xa)
    y = f(xa) - d * xa
    return (d * x) + y

# fun的图像
def test3():
    x=np.arange(0,20,0.1)
    y=fun(x)
    y2=numerical_diff(fun,x)
    y3=tangent_line(fun,5,x)

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.plot(x, y, label="f(x)")
    plt.plot(x, y2, linestyle='--', label="numerical_diff")
    plt.plot(x, y3, linestyle='--', label="tangent_line")
    plt.legend()
    plt.show()

# 计算这两个点的导数
def test4():
    print(numerical_diff(fun,5))
    print(numerical_diff(fun,10))

# 双参数function
def fun2(x):
    if x.ndim == 1:
        return np.sum(x**2)
    else:
        return np.sum(x**2, axis=1)

def _numerical_gradient_no_batch(f, x):
    h = 1e-4 # 0.0001
    grad = np.zeros_like(x)
    
    for idx in range(x.size):
        tmp_val = x[idx]
        x[idx] = float(tmp_val) + h
        fxh1 = f(x) # f(x+h)
        
        x[idx] = tmp_val - h 
        fxh2 = f(x) # f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2*h)
        
        x[idx] = tmp_val # 还原值
    return grad

def numerical_gradient(f, X):
    if X.ndim == 1:
        return _numerical_gradient_no_batch(f, X)
    else:
        grad = np.zeros_like(X)
        
        for idx, x in enumerate(X):
            grad[idx] = _numerical_gradient_no_batch(f, x)
        
        return grad

# 计算这两个点的导数
def test5():
    print(numerical_gradient(fun2,np.array([3.0,4.0])))
    print(numerical_gradient(fun2,np.array([0.0,2.0])))
    print(numerical_gradient(fun2,np.array([3.0,0.0])))

# 偏导数图形
def test6():
    x0 = np.arange(-2, 2.5, 0.25)
    x1 = np.arange(-2, 2.5, 0.25)
    X, Y = np.meshgrid(x0, x1)
    
    X = X.flatten()
    Y = Y.flatten()
    
    grad = numerical_gradient(fun2, np.array([X, Y]))
    
    plt.figure()
    plt.quiver(X, Y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([-2, 2])
    plt.ylim([-2, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

# 偏导数图形
def test7():
    x = np.array([0,1,2])
    y = np.array([0,1,2])
    grad = np.array([
        [-1,0,1],
        [0,1,0]])
    
    plt.figure()
    plt.quiver(x, y, -grad[0], -grad[1],  angles="xy",color="#666666")#,headwidth=10,scale=40,color="#444444")
    plt.xlim([0, 2])
    plt.ylim([0, 2])
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.grid()
    plt.legend()
    plt.draw()
    plt.show()

# 学习
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= lr * grad
    return x

def myf(x):
    return abs(x[0]-0.1)

def test8():
    init_x = np.array([0.2,0.1])
    y=gradient_descent(fun2, init_x=init_x, lr=0.1, step_num=100)
    print(y)

def test9():
    init_x = np.array([0.0])
    newx=gradient_descent(myf, init_x=init_x, lr=0.05, step_num=200)
    print(newx)
    print(myf(newx))


if __name__ == "__main__":
    test9()
