import numpy as np
import matplotlib.pyplot as plt

x1 = [1, 2, 3, 4, 5, 6, 7]
y1 = [1, 0, 3, 8, 5, 15, 0]

x2 = [1, 2, 3, 4, 5, 6, 7]
y2 = [10, 5, 6, 9, 20, 30, 0]


x_test=[2,0,8,6,3,6]
y_test=[5,9,4,1,6,30]

slope1=0
intercept1=0

slope2=0
intercept2=0
abline_values1=0
abline_values2=0

alpha=0.01
#print(y)
J_vals=np.matrix(np.zeros((100,2)))
for k in range(0,100):

    abline_values1 = [slope1 * i + intercept1 for i in x1]
    abline_values2 = [slope2 * i + intercept2 for i in x2]

    J1 = np.sum(np.power(np.subtract(abline_values1,y1), 2))
    J2 = np.sum(np.power(np.subtract(abline_values2, y2), 2))

    dJ_dm1=np.sum(np.dot(np.subtract(abline_values1,y1),x1))/5.0
    dJ_dc1 =np.sum(np.subtract(abline_values1,y1))/5.0

    dJ_dm2 = np.sum(np.dot(np.subtract(abline_values2, y2), x2)) / 5.0
    dJ_dc2 = np.sum(np.subtract(abline_values2, y2)) / 5.0

    slope_new1=slope1-alpha*dJ_dm1
    intercept_new1 = intercept1 - alpha * dJ_dc1

    slope_new2 = slope2 - alpha * dJ_dm2
    intercept_new2 = intercept2 - alpha * dJ_dc2

    print(J1)
    print(J2)
    slope1=slope_new1
    intercept1=intercept_new1
    slope2 = slope_new2
    intercept2 = intercept_new2

    J_vals[k,0]=J1
    J_vals[k,1]=J2

print(abline_values2)

#plt.subplot(211)

#plt.plot(x1, y1, 'ro')
#plt.plot(x1, abline_values1, 'b')
#plt.title('class1')

# plt.subplot(212)
#plt.plot(x2, y2, 'ro')
#plt.plot(x2, abline_values2, 'b')
#plt.title('class2')
#plt.show()

for i in range(0,6):
    dist1=abs(slope1*x_test[i]-y_test[i]+intercept1)/(np.sqrt(slope1**2+1+intercept1** 2))
    dist2 = abs(slope2*x_test[i]-y_test[i]+intercept2) / (np.sqrt(slope2**2+1+intercept2** 2))
    #print(dist1)
    #print(dist2)

    if dist1>dist2:
        print("2")
    else:
        print("1")



plt.plot(x1, abline_values1, 'b')
plt.plot(x2, abline_values2, 'r--')
plt.plot(x_test,y_test,'ro')

plt.show()
