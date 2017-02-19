import _pickle as cpickle
from PIL import Image
import numpy as np
import time
def unpickle(file):
    fo = open(file, 'rb')
    dict = cpickle.load(fo,encoding='latin1')
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
    dictionary = unpickle('../Datasets/cifar-10-batches-py/data_batch_4')
    [imgmat,label]=getipicnlable(dictionary,10,dodebug=0)