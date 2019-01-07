import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread

def test1():
    x=np.arange(0, 6, 0.1)
    plt.plot(x, np.cos(x),  label="cos")
    plt.plot(x, np.sin(x), linestyle="--", label="sin")
    #plt.plot(x, np.tan(x), linestyle="--", label="tan")
    plt.legend()#显示一个标签,用于标识某条线的作用
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title('sin & cos') # 标题
    plt.show()

def test2():
    plt.imshow(imread("lena.png"))
    plt.show()

if __name__ == "__main__":
    test1()