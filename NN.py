import numpy as np
import math
import matplotlib.pyplot as plt

data_1 = np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumImages5000.txt')
data_t = np.genfromtxt(r'C:\Users\jayad\Desktop\Grad\Intelligent Systems\Assignement 3\MNISTnumLabels5000.txt')
data_res = np.ascontiguousarray(data_t, dtype = np.int)

data_res_hot = np.zeros((5000, 10))
data_res_hot[np.arange(5000), data_res] = 1 
data_mat = np.concatenate((data_1, data_res_hot), axis= 1 )

w1 = np.random.rand(100, 784) * 0.01
w2 = np.random.rand(10, 100) * 0.01
w3 = np.zeros_like(w1)
w4  =np.zeros_like(w2)
eta1 = 0.1
eta2 = 0.1
error_list_train = list()
error_list_test = list()
err_list_train = []
err_list_test = []
confusion_matrix_train = np.zeros(shape = (10,10))
confusion_matrix_test = np.zeros(shape = (10,10))
for m in range(10):
 
 for v in range(10):
    correct = 0
    np.random.shuffle(data_mat)
    mat_train_dat = np.zeros(shape=(200, 794))
    mat_train_data = np.zeros(shape=(200, 784))
    mat_train_res = np.zeros(shape=(200, 10))
    ind = list(np.random.randint(0, 3999, size=200))
    for i in range(200):
        mat_train_dat[i] = data_mat[ind[i]]
    mat_train_data = mat_train_dat[:,:784]
    mat_train_res = mat_train_dat[:, 784:]
    
    
    for i in range(200):
        
        s1 = np.zeros(100)
        for a in range(100):
            s1[a] = np.sum(w1[a] * mat_train_data[i])
        h1 = [1/(1+math.exp(-x)) for x in s1]
        s2 = np.zeros(10)
        for b in range(10):
            s2[b] = np.sum(w2[b] * h1)
        yout = [1/(1+math.exp(-x)) for x in s2]
        if np.argmax(mat_train_data[i]) == np.argmax(yout): 
           correct = correct + 1
        if v ==9 and m ==9:
           confusion_matrix_train[np.argmax(yout), np.argmax(mat_train_res[i])] +=1                
        err = yout - mat_train_res[i]  
        error_list_train.append(err)
        for c in range(10):
            for d in range(100):
                w4[c, d]  =  w4[c, d]*0.1 - eta2*yout[c]*(1-yout[c])*h1[d]*err[c]
        w2 = w2+w4
        change = np.zeros(100)
        for e in range(100):
            for f in range(10):
                change[e] = change[e] + w2[f, e]*err[f]*yout[f]*(1-yout[f])
        for i1 in range(100):
            for y in range(784):
                w3[i1, y] = w3[i1, y]*0.1 - eta1 * change[i1] * h1[i1] * (1-h1[i1]) * mat_train_data[i,y]
        w1 = w1+w3
 err_list_train.append(1 - correct/200)     
    
 print("Done")

 data_test_in = data_mat[4000:,:784]
 data_test_out = data_mat[4000:, 784:]
 tp=0

 for q in range(1000):
    s_1 = np.zeros(100)
    for g in range(100):
       s_1[g] = np.sum(data_test_in[q] * w1[g])
    h1_test = [1/(1+math.exp(-x)) for x in s_1]
    s_2 = np.zeros(10)
    for h in range(10):
       s_2[h] = np.sum(h1_test * w2[h])
    out = [ 1/(1+math.exp(-x)) for x in s_2] 
    error_list_test.append(out - data_test_out[q])
    actual_val = np.argmax(data_test_out[q], axis = 0)
    obt_val = np.argmax(out, axis = 0)
    if m==9:
         
       confusion_matrix_test[obt_val, actual_val] = confusion_matrix_test[obt_val, actual_val] + 1
    if obt_val == actual_val:
       tp = tp+1 
 err_list_test.append(1 - tp/1000)
 print(tp/1000)

    


