# -*- coding: utf-8 -*-
"""
Created on Tue Sep 12 20:39:09 2017

"""
from __future__ import print_function
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from scipy.special import logsumexp
np.random.seed(6)
import pdb, traceback, sys

# load boston housing prices dataset
boston = load_boston()
x = boston['data']
N = x.shape[0]
x = np.concatenate((np.ones((506,1)),x),axis=1) #add constant one feature - no bias needed
d = x.shape[1]
y = boston['target']

idx = np.random.permutation(range(N))

#helper function
def l2(A,B):
    '''
    Input: A is a Nxd matrix
           B is a Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between A[i,:] and B[j,:]
    i.e. dist[i,j] = ||A[i,:]-B[j,:]||^2
    '''
    
    '''
    input A is 1xd
    input B is dxN
    output is 1xN
    '''
    #print('A is ')
    #print(A)
    #print('A** is')
    #print(A**2)    
    A_norm = (A**2).sum(axis=1).reshape(A.shape[0],1)
    B_norm = (B**2).sum(axis=1).reshape(1,B.shape[0])
    dist = A_norm+B_norm-2*A.dot(B.transpose())
    return dist

 
 
#to implement
def LRLS(test_datum,x_train,y_train, tau,lam=1e-5):
    '''
    Input: test_datum is a dx1 test vector
           x_train is the N_train x d design matrix
           y_train is the N_train x 1 targets vector
           tau is the local reweighting parameter
           lam is the regularization parameter
    output is y_hat the prediction on test_datum
    '''
    
    d = x_train.shape[1]
    distance = l2(test_datum.T, x_train)    

    Ai = -distance.T/(2*tau**2)
    B = np.max(Ai)
    weight_numerator = np.exp(Ai-B)
    weight_denominator = np.exp(logsumexp(Ai-B))    
    a = weight_numerator/weight_denominator
    A = np.diag(a[:,0])
    Q = np.dot(x_train.T, np.dot(A, x_train)) + lam * np.eye(d)
    b = np.dot(x_train.T, np.dot(A, y_train))
    
    wStar = np.linalg.solve(Q, b)
    
    y = np.dot(test_datum.T,wStar)
        
    return y

def compute_average_loss(test_data, test_labels, x_train, y_train, taus, lam=1e-5):
    '''
    This function computes the average loss for a given dataset, as described by
    question 2(c)
    '''
    losses = []
    for tau in taus:
        total_loss = 0
        for test_datum,test_label in zip(test_data, test_labels):
            test_datum = test_datum.reshape([d, 1])
            predicted_label = LRLS(test_datum, x_train, y_train, tau, lam)
            loss = (1/2)*(predicted_label - test_label)**2
            total_loss += loss
        average_loss = total_loss/len(y_train)
        losses.append(average_loss)
    
    return losses


def run_validation(x,y,taus,val_frac):
    '''
    Input: x is the N x d design matrix
           y is the N x 1 targets vector    
           taus is a vector of tau values to evaluate
           val_frac is the fraction of examples to use as validation data
    output is
           a vector of training losses, one for each tau value
           a vector of validation losses, one for each tau value
    '''
    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=77)    
    
    train_losses = compute_average_loss(X_train, y_train, X_train, y_train, taus)
    test_losses = compute_average_loss(X_train, y_train, X_test, y_test, taus)
    
    
    return train_losses, test_losses



############## HELPER FUNCTION #####################
    


if __name__ == "__main__":
    # In this excersice we fixed lambda (hard coded to 1e-5) and only set tau value. Feel free to play with lambda as well if you wish
    taus = np.logspace(1.0,3,200)
    train_losses, test_losses = run_validation(x,y,taus,val_frac=0.3)
    plt.semilogx(train_losses)
    plt.semilogx(test_losses)
