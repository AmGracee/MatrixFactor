import numpy as np
import matplotlib.pyplot as plt


def LFM_2(a, k, iter_times, alpha=0.01, lr=0.01):
    '''
    :param a: 需要分解的矩阵
    :param k: 分解的隐变量个数
    :param iter_times: 迭代次数
    :param alpha: 正则系数
    :param lr: 学习速率
    :return: 分解完毕的矩阵，u，v，以及误差列表err_list
    '''
    assert type(a) == np.ndarray
    m,n = a.shape
    u = np.random.rand(m,k)
    v = np.random.randn(k,n)
    err_list = []

    for t in range(iter_times):
        A = np.matmul(u,v)
        err = a - A
        grad_u = -2 * np.matmul(err, v.transpose()) + 2 * alpha * u
        grad_v = -2 * np.matmul(u.transpose(), err) + 2 * alpha * v
        u -= lr * grad_u
        v -= lr * grad_v

        err2 = np.multiply(err,err)
        err2_sum = np.sum(np.sum(err2))
        err_list.append(err2_sum)
    return u,v,err_list
A = np.array([[5,5,0,5],[5,0,3,4],[3,4,0,3],[0,0,5,3],[5,4,4,5],[5,4,5,5]])
U,V,err_list = LFM_2(A,3,iter_times=200, lr=0.01,alpha=0.01)

err_log = np.log(np.array(err_list)) #log函数

plt.plot("err_list")
plt.plot(err_list)
plt.figure(3) # 画两张图
plt.plot(err_log)
plt.show()
print("U=",U)
print("V=",V)
# print("ERR=",err_list)
print(err_list[-1]) #输出最后一列