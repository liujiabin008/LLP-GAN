import numpy as np
import scipy as sp
import scipy.io as sio
from scipy.misc import *
from keras.utils import to_categorical

#Dataset location
train_location = 'dataset/SVHN/train_32x32.mat'
test_location = 'dataset/SVHN/test_32x32.mat'



def load_train_data():
    train_dict = sio.loadmat(train_location)
    X = np.asarray(train_dict['X'])

    X_train = []
    for i in xrange(X.shape[3]):
        X_train.append(X[:,:,:,i])
    X_train = np.asarray(X_train)

    Y_train = train_dict['y']
    for i in xrange(len(Y_train)):
        if Y_train[i]%10 == 0:
            Y_train[i] = 0
    Y_train = to_categorical(Y_train,10)
    return (X_train,Y_train)

def load_test_data():
    test_dict = sio.loadmat(test_location)
    X = np.asarray(test_dict['X'])

    X_test = []
    for i in xrange(X.shape[3]):
        X_test.append(X[:,:,:,i])
    X_test = np.asarray(X_test)

    Y_test = test_dict['y']
    for i in xrange(len(Y_test)):
        if Y_test[i]%10 == 0:
            Y_test[i] = 0

    Y_test = to_categorical(Y_test,10)
    #print Y_test[0:30,:]
    #return (X_test[0:5000,:],Y_test[0:5000,:])
    return (X_test,Y_test)


if __name__ == "__main__":
    load_test_data()