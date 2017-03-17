import numpy as np
import copy
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

def removemean(data,mean,no_of_examp):
    data_rm=copy.deepcopy(data)
    for i in range(no_of_examp):
        data_rm[i,:]=data_rm[i,:]-mean
    return data_rm

data=np.matrix(np.random.random((100,2)))+0.2
data=np.concatenate((data,np.matrix(np.random.random((100,2)))+0.8))
(no_of_examp,no_of_feat)=np.shape(data)
no_of_comp=5
# Random Initialization
meaninitpos=np.random.randint(1,no_of_examp,no_of_comp)
weights=1/no_of_comp*np.ones(no_of_comp)
means=np.matrix(data[meaninitpos,:])
cov=np.zeros((no_of_comp,no_of_feat,no_of_feat))
for i in range(0,no_of_comp):
    cov[i,:,:]=np.matrix(np.diag(np.random.random(no_of_feat)+0.1))

num_iterations=20
import time
s=time.time()
for i in range(num_iterations):
    # Expect
    print(i)
    liklihood=np.zeros((no_of_examp,no_of_comp))
    for i in range(0,no_of_comp):
        for j in range(0,no_of_examp):
            X=np.matrix(data[j,:]).T
            mu=np.matrix(means[i]).T
            C=np.matrix(cov[i])
            Cinv=np.linalg.inv(C)
            Cdet=np.linalg.det(C)
            liklihood[j,i]=weights[i]*1/(np.sqrt((2*np.pi)**no_of_feat*Cdet))*np.exp(-(1/2*((X-mu).T*Cinv*(X-mu))))
    prob=np.zeros((no_of_examp,no_of_comp))
    gamma=np.zeros((no_of_examp,no_of_comp))
    for i in range(0,no_of_examp):
        prob[i,:]=liklihood[i,:]
        prob[i,np.argmax(liklihood[i,:])]=0
        prob[i,:]/=np.max(liklihood[i,:])
        prob[i,np.argmax(liklihood[i,:])]=1

    for i in range(0,no_of_comp):
        for j in range(0, no_of_examp):
            gamma[j,i]=prob[j,i]/np.sum(prob[j,:])
    # Maximize
    for i in range(no_of_comp):
        # Mean
        means[i,:]=np.dot(data.T,gamma[:,i])/np.sum(gamma[:,i])
        # Cov
        data_rm=removemean(data,means[i,:],no_of_examp)
        diagnolw=np.diag(gamma[:,i])
        cov[i,:,:]=np.diag(np.diag(data_rm.T*diagnolw*data_rm/np.sum(gamma[:,i])))
        # Weight
        weights[i]=1/no_of_comp*np.sum(gamma[:,i])


print(time.time()-s)
print(means)
print(weights)
print(cov)
x = np.arange(0, np.max(data), 0.025)
y = np.arange(0, np.max(data), 0.025)
X, Y = np.meshgrid(x, y)
Z=np.zeros((no_of_comp,len(x),len(y)))
for i in range(no_of_comp):
    Z[i]=mlab.bivariate_normal(X,Y,np.sqrt(cov[i][0,0]),np.sqrt(cov[i][0,0]),means[i,0],means[i,1],cov[i][1,0])

plt.contour(X,Y,np.dot(Z.T,weights))
plt.scatter(data[:,0],data[:,1])
plt.show()
import pickle
f=open('Data.pckl','wb')
pickle.dump((means,cov,weights,data),f)
