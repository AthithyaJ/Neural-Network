import numpy as np
import math


def Weight_change(mat_train, mat_train_res, w1, w2):
    import numpy as np
    s = np.zeros(100)
    s2 = np.zeros(10)
    h = np.zeros(100)
    D1 = np.zeros(100)
    D2 = np.zeros(10)
    y = np.zeros(10)
    e = np.zeros(10)
    eta1 = 0.01
    eta2 = 0.02
    moment = 0.1
    r = 100
    c = 784
    r1 = 10
    c1 = 100
    w3 = np.random.rand(10, 100)
    w4 = np.random.rand(100, 784)
    for n in range(10):
     for m in range(4000):
        for i in range(r):
            s[i] = np.dot(w1[i], mat_train[m])
            h[i] = 1 / (1 + math.exp(-s[i]))
            D1[i] = h[i] * (1 - h[i])
        for k in range(10):
            s2[k] = np.dot(w2[k], h)
            y[k] = 1 / (1 + math.exp(-s2[k]))
            D2[k] = y[k] * (1 - y[k])
        e = np.array(y - mat_train_res[m])
        for a in range(r1):
            for b in range(c1):
                w3[a][b] = w2[a][b]*moment - eta2 * D2[a] * h[a] * e[a]
        for i in range(r):
            chan = 0
            for k in range(r1):
                chan += w3[k][i] * D2[k] * e[k]
            for j in range(c):
                w4[i][j] = w1[i][j]*moment + eta1 * chan * D1[i] * mat_train[m][j]
    return w3, w4


data_1 = np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumImages5000.txt')
data_t = np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumLabels5000.txt')
w1 = np.random.rand(100, 784) * 0.001
w2 = np.random.rand(10, 100) * 0.001
data_t1 = np.zeros((5000, 10))
data_t = np.ascontiguousarray(data_t, dtype=np.int)
data_t1[np.arange(5000), data_t] = 1
mat = np.hstack((data_1, data_t1))
np.random.shuffle(mat)
mat_train = mat[:4000, :784]
mat_train_res = mat[:4000, 784:]

W1, W2 = Weight_change(mat_train, mat_train_res, w1, w2)
