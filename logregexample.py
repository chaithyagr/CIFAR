#logistic regression example for 2D data
import numpy as np
import sys
import time

start_time = time.time()

#initialising data
x1 = np.array([[0,0],[0,1],[2,2],[2,3],[-1,-1],[1,1]])
n = x1.shape[0]
y = [0,0,1,1,0,1]
t = x1.shape[1] + 1
theta = np.zeros(t)
one = np.array([1])
x= np.zeros(n*t).reshape(n,t)

#finding decision boundary using sigmoid function
def sigmoid(z):
	g = 1/(1+np.exp(-z))
	if g>0.5:
		return 1
	else:
		return 0

#finding derivative of error function
while(1):
	j = np.zeros(t)
	for i in range(0,n):
		x[i] = np.concatenate((one,x1[i]))
		w = np.dot(theta,x[i])
		p = sigmoid(w)
		e = p - y[i]
		j[0] = x[i][0]*e+j[0]
		j[1] = x[i][1]*e+j[1]
		j[2] = x[i][2]*e+j[2]
			
	#taking alpha = 0.03
	for i in range(0,t):
		J = j[i]/n
		theta[i] = theta[i] - 0.03*J

	if (j[0]==0.0) and (j[1]==0.0) and (j[2]==0.0):
		print 'Obtained values of theta are \n',theta,'\n'
		break

print("--- Execution time is %s seconds ---" % (time.time() - start_time))
