import math

import numpy as np

# 定义矩阵分解函数
def lfm(a,k):
    '''
    :param a: 表示需要分解的矩阵
    :param k: 表示分解的属性（隐变量）个数
    '''
    assert  type(a) == np.ndarray
    m,n = a.shape
    alpha = 0.01
    lambda_ = 0.01
    u = np.random.rand(m,k)
    v = np.random.randn(k,n)
    for t in range(1000):
        for i in range(m):
            for j in range(n):
                if math.fabs(a[i][j]) > 1e-4:
                    err = a[i][j] - np.dot(u[i],v[:,j])
                    for r in range(k):
                        grad_u = err * v[r][j] - lambda_ * u[i][r]  #对u求导
                        grad_v = err * u[i][r] - lambda_ * v[r][j]  #对v求导
                        u[i][r] += alpha * grad_u   #u的梯度更新
                        v[r][j] += alpha * grad_v   #v梯度更新
    return u,v


A = np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
b,c = lfm(A,3)

#查看“商品—商品属性”矩阵b和“商品属性—客户喜好”矩阵c
print(b)
print(c)
print(np.dot(b,c))  #可以看出之前A为0的，还原后有了预测值，

