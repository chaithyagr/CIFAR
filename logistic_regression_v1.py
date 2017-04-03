import numpy as np
import matplotlib.pyplot as plt

x1=np.matrix(np.random.randint(1,10,size=(5,9)))
y1=[1,0,0,0,1]
#print(y1[2])
one=np.matrix(np.ones((5,1)))
X1=np.append(x1,one,axis=1)
#print(X1)
#print(x1)
#print(y1)

#defining parameters- theta[10]
theta=np.matrix(np.zeros((1,10)))
#print(theta)
J_I=np.matrix((np.zeros((1,5))))
d_J_d_theta=np.array(np.zeros((10,1)))
alpha=0.0001
#doing 50 iterations of gradient descent
for k in range(0,500):

    J=0
    z=np.dot(theta,X1.T)
    h=1/(1+np.exp(-z))

    h = np.asarray(h).reshape(-1)

    J_I=(np.dot(y1,np.log(h))+(np.dot(np.subtract(1,y1),np.log(np.subtract(1,h)))))/5.0
    #print(J_I)
    J=np.sum(J_I)
    J=J*-1

    #print(J)
    d_J_d_theta[0]=np.sum(np.subtract(h,y1))/5.0
    #print(np.subtract(h,y1))
    for i in range (1,10):
        d_J_d_theta[i]=np.sum(np.dot(np.subtract(h,y1),x1[:,i-1]))/5.0


    theta_new=theta-alpha*d_J_d_theta.T

    #print(d_J_d_theta)
    #print(x1)
    #print(theta_new)

    theta=theta_new
    print(J)
    print(theta)

