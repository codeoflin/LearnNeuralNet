import numpy as np

# 这里连续进行了4次AND门和OR门运算
def test1():
    x=np.array([[0,0],[0,1],[1,0],[1,1]])#输入层,4组输入

    #隐含层 2个神经元
    #第一个神经元的权重分别是0.5 0.5
    #第二个神经元的权重分别是0.3 0.3
    w=np.array([[0.5,0.3],[0.5,0.3]])

    #偏置
    b=np.array([-0.7,-0.2])

    y=np.dot(x,w)+b
    
    # AND输出
    andy=(y[:,0]>0).astype(np.int)#转为int
    print(andy)

    # OR输出
    ory=(y[:,1]>0).astype(np.int)#转为int
    print(ory)

if __name__=="__main__":
    test1()