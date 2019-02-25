import numpy as np
import math

H=1e-4

# 导数
def gradient(f,x):
    return (f(x+H)-f(x-H))/(H*2)

# 偏导数
def numerical_gradient(f,x):
    grad=np.zeros_like(x)
    for idx in range(x.size):
        tmp=x[idx]
        
        x[idx]=tmp+H
        y1=f(x)

        x[idx]=tmp-H
        y2=f(x)

        x[idx]=tmp
        grad[idx]=(y1-y2)/(H*2)

    return grad

# 损失函数
# 损失函数的作用是用来计算 一个函数的Y 与 目标答案 的距离
# 值域: 0表示完全符合目标答案 数字越大表示与目标答案越远
def lose(f):
    return lambda x:abs(f(x))

def test1():
    f1=lambda x:x+1         # gradient=+1
    f2=lambda x:f1(x)**2    # gradient=2(x+1)
    f3=lambda x:f2(f1(x))+2 # gradient=2(x+2)
    x=100
    print(gradient(f1,x))   # 1
    print(gradient(f2,x))   # 202
    print(gradient(f3,x))   # 204

    print(f1(x))            # 101
    print(f2(x))            # 10201
    print(f3(x))            # 10405
    pass

# 1/x的求导过程
def test2():
    # 要求导的f
    f1=lambda x:1.0/x

    # 反向求导
    d1_f1=lambda x:-(f1(x)**2)

    # 微分求导
    d2_f1=lambda x:(f1(x+H)-f1(x-H))/(H*2)

    # 微分求导2
    d3_f1=lambda x:(f1(x+H)-f1(x))/H


    print(d1_f1(1)) # -1
    print(d2_f1(1)) # -1

    print(d1_f1(2)) # -0.25
    print(d2_f1(2)) # -0.25

    print(d1_f1(3)) # -0.1111
    print(d2_f1(3)) # -0.1111

def test3():
    # 要求导的f
    f1=lambda x:np.exp(x)

    # 反向求导
    d1_f1=lambda x:np.exp(x)

    # 微分求导
    d2_f1=lambda x:(f1(x+H)-f1(x-H))/(H*2)

    print(d1_f1(1)) # 2.718281828459045
    print(d2_f1(1)) # 2.718281832989611

    print(d1_f1(2)) # 7.38905609893065
    print(d2_f1(2)) # 7.389056111253289

    print(d1_f1(3)) # 20.085536923187668
    print(d2_f1(3)) # 20.085536956706562



if(__name__=="__main__"):
    test3()