import numpy as np
import matplotlib as plt

train_data=np.matrix(np.load('../gmm.py/files/sorted_data_lda.npy'))
train_label=np.array(np.load('../gmm.py/files/sorted_labels.npy'))
test_data=np.matrix(np.load('../gmm.py/files/test_data_lda.npy'))
test_label=np.array(np.load('../gmm.py/files/test_label.npy'))

theta_final = np.matrix(np.zeros((10, 10)))
theta=np.matrix(np.zeros((1,10)))
#J_I = np.matrix((np.zeros((10, 50000))))
d_J_d_theta = np.zeros((10, 1))
alpha = 0.0001
theta_new=np.matrix(np.zeros((1, 10)))



for i in range(0,10):

    label = np.zeros((50000, 1))
    x=train_data
    label[5000*i:(i+1)*5000]=1

    one = np.matrix(np.ones((50000, 1)))
    X1 = np.append(x, one, axis=1)
    # doing 50 iterations of gradient descent
    for k in range(0, 50):

        J = 0
        z = np.dot(theta[i], X1.T)
        h = 1 / (1 + np.exp(-z))

        #h = np.asarray(h).reshape(-1)
        label = np.asarray(label).reshape(-1)

        h=np.resize(h,(50000,1))
        print(np.shape(h))
        print(np.shape(label))


        J = np.sum(np.dot(label, np.log(h)) + (np.dot(np.subtract(1, label), np.log(np.subtract(1, h))))) / 50000.0
        J = J* -1
        print(J)

        d_J_d_theta[0] = np.sum(np.subtract(h,label))/50000.0
        for j in range(1, 10):
            d_J_d_theta[j] = np.sum(np.dot(np.subtract(h, label), train_data[:, j - 1])) / 5.0

        theta_new = theta - alpha * d_J_d_theta.T
        theta = theta_new

    theta_final[i,:]=theta

    #obtained paramerts : theta[]


#to classify
z1=np.zeros((10,1))

new_label=np.zeros((10000,1))
for i in range(0,10000):

    X1=train_data[i]
    one = np.matrix(np.ones((50000, 1)))
    X1 = np.append(X1, one, axis=1)

    for k in range(0,10):

        z1[k,:]=np.dot(theta[i], X1.T)
        h[k,:]=1/(1+np.exp(-z1[k]))

        if h[k]>0.5:
            new_label[i]=k


    if new_label[i]==test_label[i]:

        c=c+1

accuracy=c/10000



        

