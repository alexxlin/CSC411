import random
import numpy as np
from sklearn.datasets import load_boston

def gradient_descent(X, y, lr, num_iter, delta):
    
    w_b = np.zeros(X.shape[1])
    N = X.shape[0]
    
    for i in range(num_iter):
        Hprime = compute_Hprime(np.dot(X, w_b) - y, delta)
        w_b = w_b - lr * (1/N)*np.dot(X.T, Hprime)
        L = compute_loss(np.dot(X, w_b)-y, delta)
        J = (1/N)*np.sum(L)
        print("the total loss is " + str(J))
        #print("The parameter for iteration " + str(i) + " is " + str(w_b))
    return w_b

def compute_Hprime(a, delta):
    
	Hprime = np.copy(a)

	index1 = np.where(np.absolute(a) <= delta)
	index2 = np.where(a > delta)
	index3 = np.where(a < -delta)

	Hprime[index1] = a[index1]
	Hprime[index2] = delta
	Hprime[index3] = -delta

	return Hprime

# compute the huber loss
def compute_loss(a, delta):

	loss = np.copy(a)

	index1 = np.where(np.absolute(a) <= delta)
	index2 = np.where(a > delta)
	index3 = np.where(a < -delta)

	loss[index1] = (1/2)*np.power((a)[index1],2)
	#loss[index2] = delta*(np.absolute((a)[y_idx_2])-0.5*delta)
	loss[index2] = delta*(a[index2]-0.5*delta)
	loss[index3] = delta*(-a[index3]-0.5*delta)

	return loss    


if __name__ == "__main__":
    boston = load_boston()
    X = boston.data
    
    sampleSize = len(X)
    parameterSize = len(X[0])
    
    #add X0 to the X input matrix, all X0 have value of 1
    #This will transform y = wX + b into y = wX
    X = np.hstack((X, np.zeros((X.shape[0], 1), dtype=X.dtype)))
    
    y = boston.target
    
    lr = 0.00001
    
    num_iter = 100
    
    delta = 2
 
    t1 = gradient_descent(X, y, lr, num_iter , delta)