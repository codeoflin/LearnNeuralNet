import numpy as np

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
def lose(f):
    return lambda x:abs(f(x))

# 试一下单参数func梯度下降
def test1():
    f=lambda x:x-0.1
    learnrate=0.01
    times=5000
    x=1.0
    for i in range(times):
        grad=gradient(lose(f),x)
        x = x - grad*learnrate
        
    print(x)
    print(gradient(lose,x))
    print(f(x))
    pass

# 试一下多参数func梯度下降
def test2():
    learnrate=0.001
    times=10000
    f=lambda x:(x[0]**2)+(x[1]*2)+np.sum(x)-50
    x=np.array([1.0,0.0])
    for i in range(times):
        grad = numerical_gradient(lose(f),x)
        x = x - grad*learnrate
    print(x)
    print(gradient(lose(f),x))
    print(f(x))

if(__name__=="__main__"):
    test2()