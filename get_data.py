import cPickle as pickle
#from PIL import Image
import numpy as np
import math
import time
def unpickle(file):
    fo = open(file, 'rb')

    dict = pickle.load(open(file, "rb"))
    fo.close()
    return dict

def getipicnlable(dictionary,i,dodebug):
    img=dictionary['data'][i]
    imgmat=np.reshape(img,(32,32,3),order='F')
    imgmat=np.rot90(imgmat,-1)
    label=dictionary['labels'][i]
    if(dodebug==1):
        print(label)
        img = Image.fromarray(imgmat, 'RGB')
        img.show()
    return (imgmat, label)


if __name__ == '__main__':
    dictionary = unpickle('../gmm.py/Datasets/cifar-10-batches-py/data_batch_1')
    feature1=np.zeros((10000,32,32,3))
    feature2=np.zeros((10000,32,32,3))
    feature3 = np.zeros((10000, 32, 32, 3))
    feature4 = np.zeros((10000, 32, 32, 3))

    for i in range(0,10000):

        [imgmat,label]=getipicnlable(dictionary,i,dodebug=0)

        feature1[i,:,:,:]=np.log(imgmat)

        a,b,c=np.gradient(imgmat)

        feature2[i,:,:,:]=a
        feature3[i, :, :, :] = b
        feature4[i, :, :, :] =c

    