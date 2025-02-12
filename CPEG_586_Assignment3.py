import os
import sys
import cv2
import numpy as np
from sklearn.utils import shuffle
import math
from MyEnums import TrainingType
from MyEnums import ActivationType
from MyEnums import BatchNormMode
from MyEnums import LROptimizerType

trainingImages = len(os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Training1000"))
testImages = len(os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Test10000"))
train = np.empty((trainingImages,28,28),dtype='float64')
trainY = np.zeros((trainingImages,10,1))
test = np.empty((testImages,28,28),dtype='float64')
testY = np.zeros((testImages,10,1))

trainType = TrainingType.MiniBatch
LROptType = LROptimizerType.ADAM
batchSize = 10.0
Epsillon = 1E-8


HiddenLayer1Neurons = 75
OutputLayerNeurons = 10
numEpochs = 1000

#load images
i = 0
for filename in os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Training1000"):
    y = int(filename[0])
    trainY[i,y] = 1.0
    train[i] = cv2.imread('S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Training1000\\{0}'.format(filename),0)/255.0 #for color use 1
    i += 1

j = 0
for filename in os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Test10000"):
    y = int(filename[0])
    testY[j,y] = 1.0
    test[j] = cv2.imread('S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\Data\\Test10000\\{0}'.format(filename),0)/255.0 
    j += 1

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2])
testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2])




#get array ready for 80% dropout
dropoutZeros = (int)(0.2*HiddenLayer1Neurons)
dropoutOnes = HiddenLayer1Neurons - dropoutZeros
ZerosVector = np.zeros((dropoutZeros,1))
OnesVector = np.ones((dropoutOnes,1))
HiddenLayer1Dropout = np.concatenate((ZerosVector,OnesVector))

w1 = np.random.uniform(low=-0.1,high=0.1,size=(HiddenLayer1Neurons,784))
b1 = np.random.uniform(low=-1.0,high=1.0,size=(HiddenLayer1Neurons,1))
w2 = np.random.uniform(low=-0.1,high=0.1,size=(OutputLayerNeurons,HiddenLayer1Neurons))
b2 = np.random.uniform(low=-1.0,high=1.0,size=(OutputLayerNeurons,1))
gradW2 = np.zeros((w2.shape))
gradB2 = np.zeros((b2.shape))
gradW1 = np.zeros((w1.shape))
gradB1 = np.zeros((b1.shape))
learningRate = 0.01

print("End of Setup.\n")


def Sigmoid(s):
    return 1.0/(1.0+np.exp(-1.0*s))

def TanH(s):
    return np.tanh(s)

def Relu(s):
    return np.maximum(0,s)

def Softmax(s):
    if (s.shape[0] == s.size):
        ex = np.exp(s)
        return ex/ex.sum()
    ex = np.exp(s)
    for i in range (ex.shape[0]):
        denom = ex[i,:].sum()
        ex[i,:] = ex[i,:]/denom
        return ex

def logLoss(a,y):
    if len(a) != len(y):
        raise(IndexError("logLoss: The size of a and y have to be the same."))
    resLoss = 0
    for z in range(0,len(a)):
        resLoss += -(y[z,0] * np.log(a[z,0]) + (1.0 - y[z,0]) * np.log(1.0 - a[z,0]))
    return resLoss

#For Adam
mtw1 = np.zeros((w1.shape))
mtb1 = np.zeros((b1.shape))
vtw1 = np.zeros((w1.shape))
vtb1 = np.zeros((b1.shape))

mtw2 = np.zeros((w2.shape))
mtb2 = np.zeros((b2.shape))
vtw2 = np.zeros((w2.shape))
vtb2 = np.zeros((b2.shape))


def Adam(gradientW,gradientB,mtw,mtb,vtw,vtb,Beta1 = 0.9,Beta2 = 0.999):
    mtw = Beta1 * mtw + (1 - Beta1)*gradientW
    mtb = Beta1 * mtb + (1 - Beta1)*gradientB
    vtw = Beta2 * vtw + (1 - Beta2)*gradientW*gradientW
    vtb = Beta2 * vtb + (1 - Beta2)*gradientB*gradientB

    mtwhat = mtw / (1 - Beta1)
    mtbhat = mtb / (1 - Beta1)
    vtwhat = vtw / (1 - Beta2)
    vtbhat = vtb / (1 - Beta2)

    return mtwhat,mtbhat,vtwhat,vtbhat


for ep in range(0,numEpochs):
    # Shuffle data between before each Epoch.
    trainX,trainY = shuffle(trainX, trainY)
    loss = 0
    HiddenLayer1Dropout = shuffle(HiddenLayer1Dropout)

    for i in range(0,trainingImages):
        #Forward Pass
        currImg = trainX[i]
        currImg = currImg.reshape(len(currImg),1)
        s1 = np.dot(w1,currImg) + b1
        a1 = Sigmoid(s1) * HiddenLayer1Dropout
        s2 = np.dot(w2,a1) + b2
        a2 = Sigmoid(s2)
        
        #loss = logLoss(a2,trainY[i])
        loss += (0.5*((a2 - trainY[i])*(a2 - trainY[i]))).sum()
        # if i<100:
        #     print("Loss: " + str(loss))

        # Back Propagation
        yma2 = trainY[i] - a2 # y minus a2
        aoma2 = a2*(1.0 - a2) #a2 * one minus a2
        delta2 = ((yma2) * (aoma2)).T
        aoma = a1*(1.0 - a1) # a1 * one minus a1
        delta1 = (np.dot(delta2,w2)).T * aoma # (((delta2).dot(w2)).T * (aoma)) 


        #update weights and biases
        if trainType == TrainingType.Stochastic:
            gradW2 = np.dot(a1, delta2).T #(delta2.T).dot(a1.T)
            gradB2 = delta2.T
            gradW1 = np.dot(delta1,currImg.T) #(delta1).dot(currImg.T)
            gradB1 = delta1
            
            w1 = w1 - learningRate * -1.0*gradW1
            w2 = w2 - learningRate * -1.0*gradW2
            b1 = b1 - learningRate * -1.0*gradB1
            b2 = b2 - learningRate * -1.0*gradB2
        elif trainType == TrainingType.MiniBatch:
            gradW2 += (np.dot(a1, delta2).T) 
            gradB2 += (delta2.T)
            gradW1 += (np.dot(delta1,currImg.T))
            gradB1 += delta1
            #if you've hit a full batch, update weights and biases.
            if i % batchSize == (batchSize - 1):
                mtwhat1, mtbhat1, vtwhat1, vtbhat1 = Adam(gradW1,gradB1,mtw1,mtb1,vtw1,vtb1)
                mtwhat2, mtbhat2, vtwhat2, vtbhat2 = Adam(gradW2,gradB2,mtw2,mtb2,vtw2,vtb2)
                w1 = w1 - learningRate * (1/batchSize) * mtwhat1 * -1.0 /((vtwhat1**0.5) + Epsillon)
                b1 = b1 - learningRate * (1/batchSize) * mtbhat1 * -1.0 /((vtbhat1**0.5) + Epsillon)
                w2 = w2 - learningRate * (1/batchSize) * mtwhat2 * -1.0 /((vtwhat2**0.5) + Epsillon)
                b2 = b2 - learningRate * (1/batchSize) * mtbhat2 * -1.0 /((vtbhat2**0.5) + Epsillon)
                # w1 = w1 - learningRate * -1.0*(gradW1/batchSize)
                # w2 = w2 - learningRate * -1.0*(gradW2/batchSize)
                # b1 = b1 - learningRate * -1.0*(gradB1/batchSize)
                # b2 = b2 - learningRate * -1.0*(gradB2/batchSize)
                gradW2 = np.zeros((w2.shape))
                gradB2 = np.zeros((b2.shape))
                gradW1 = np.zeros((w1.shape))
                gradB1 = np.zeros((b1.shape))

        #print("New w1, w2, b1, and b2: \n" + str(w1) +  " \n" +  str(w2) + " \n" + str(b1) + " \n" +str(b2) + " \n")
        
        # if i%100 == 0:
        #     print("Epoch: " + str(ep) + "   Round: " + str(i) + "     Loss: " + str(loss))
            #print("Training Y is: " + str(trainY[i]))
            #print(str(loss))
    print("epoch = " + str(ep) + " loss = " + (str(loss))) 

print("done training , starting testing..")
accuracyCount = 0
for i in range(testY.shape[0]):
    # do forward pass
    currImg = testX[i]
    currImg = currImg.reshape(len(currImg),1)
    s1 = np.dot(w1,currImg) + b1
    a1 = 1.0/(1.0+np.exp(-1.0*s1)) # np.exp operates on the array
    s2 = np.dot(w2,a1) + b2
    a2 = 1.0/(1.0+np.exp(-1.0*s2))
    # determine index of maximum output value
    a2index = a2.argmax(axis = 0)
    currY = testY[i]
    # print(str(currY))
    # if (testY[i,a2index,0] == 1):
    if (currY[a2index,0] == 1):
        accuracyCount = accuracyCount + 1
    else:
        print("The actual result of %i is not the expected result of %i" %(a2index,currY.argmax(axis = 0)))
print("Accuracy count = " + str(100*accuracyCount/testImages) + "%")