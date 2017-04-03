import numpy as np
import time

start_time = time.time()

#loading data from previously stored files
x1 = np.load('xtrain_lda.npy')
y1 = np.load('ytrain.npy')
xtest1 = np.load('xtest_lda.npy')
ytest = np.load('ytest.npy')

#initialising variables
n = x1.shape[0]
nt = xtest1.shape[0]
t = x1.shape[1] + 1
one = np.array([1])
x= np.zeros(n*t).reshape(n,t)
xtest = np.zeros(nt*t).reshape(nt,t)

g0 = np.zeros(n*t).reshape(n,t)
gt = np.zeros(nt*t).reshape(nt,t)

#sigmoid function
def sigmoid(z):
	g = 1/(1+np.exp(-z))
	return g

for i in range(0,n):
	x[i] = np.concatenate((one,x1[i]))

for i in range(0,nt):
	xtest[i] = np.concatenate((one,xtest1[i]))

def findtheta(x,y1,xtest,ytest,n,t,c):
	theta = np.zeros(t)	
	nt = xtest.shape[0]
	while(1):
		#finding derivative of error function
		j = np.zeros(t)
		for i in range(0,n):
			w = np.dot(theta,x[i])
			g = sigmoid(w)
			#prediction
			if g>0.5:
				p = 1
			else:
				p = 0
			#actual y
			if y1[i]==c:
				y = 1
			else:
				y = 0
			#error	
			e = p - y
			for k in range(0,t):
				j[k] = x[i][k]*e+j[k]
				
		#taking alpha = 0.03 and setting some error threshold
		tab = 0
		for i in range(0,t):
			J = j[i]/n
			#print J
			theta[i] = theta[i] - 0.3*J #.03 for batch1 works well
			if abs(J)>.04: #.02 works well for batch1. .06 for training data
				tab = 1
		
		if tab == 0:
			gf = np.zeros(n)
			gtest = np.zeros(nt)
			
			for i in range(0,n):
				#finding probability for class c with the optimal theta values for same training data
				wf = np.dot(theta,x[i])
				gf[i] = sigmoid(wf)

			for i in range(0,nt):
				#finding probability for class c for test data
				wtest = np.dot(theta,xtest[i])
				gtest[i] = sigmoid(wtest)

			return gf,gtest

#obtaining G values for training and test data
g0[:,0],gt[:,0] = findtheta(x,y1,xtest,ytest,n,t,0)
print '1 done\n'
g0[:,1],gt[:,1] = findtheta(x,y1,xtest,ytest,n,t,1)
print '2 done\n'
g0[:,2],gt[:,2] = findtheta(x,y1,xtest,ytest,n,t,2)
print '3 done\n'
g0[:,3],gt[:,3] = findtheta(x,y1,xtest,ytest,n,t,3)
print '4\n'
g0[:,4],gt[:,4] = findtheta(x,y1,xtest,ytest,n,t,4)
print '5\n'
g0[:,5],gt[:,5] = findtheta(x,y1,xtest,ytest,n,t,5)
print '6\n'
g0[:,6],gt[:,6] = findtheta(x,y1,xtest,ytest,n,t,6)
print '7\n'
g0[:,7],gt[:,7] = findtheta(x,y1,xtest,ytest,n,t,7)
print '8\n'
g0[:,8],gt[:,8] = findtheta(x,y1,xtest,ytest,n,t,8)
print '9\n'
g0[:,9],gt[:,9] = findtheta(x,y1,xtest,ytest,n,t,9)
print '10\n'

#finding class corresponding to max probability from sigmoid function
def findclass(g,y,n):
	t = 0
	for i in range(n):
		cl = np.argmax(g[i])
		if cl==y[i]:
			t = t +1
		else:
			continue
	return t

t = findclass(g0,y1,n)
print 'Number of true predictions on training data are ',t,'\n'
pred = (float(t)/float(n))*100
print 'Prediction accuracy is ',pred,'%\n'

t = findclass(gt,ytest,nt)
print 'Number of true predictions on test data are ',t,'\n'
pred = (float(t)/float(nt))*100
print 'Prediction accuracy is ',pred,'%\n'

print("--- Execution time is %s seconds ---" % (time.time() - start_time))