import os
import sys
import cv2
import numpy as np
from sklearn.utils import shuffle
import math

trainingImages = len(os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Training1000"))
testImages = len(os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Test10000"))
train = np.empty((trainingImages,28,28),dtype='float64')
trainY = np.zeros((trainingImages,10,1))
test = np.empty((testImages,28,28),dtype='float64')
testY = np.zeros((testImages,10,1))

#load images
i = 0
for filename in os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Training1000"):
    y = int(filename[0])
    trainY[i,y] = 1.0
    train[i] = cv2.imread('S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Training1000\\{0}'.format(filename),0)/255.0 #for color use 1
    i += 1

j = 0
for filename in os.listdir("S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Test10000"):
    y = int(filename[0])
    testY[j,y] = 1.0
    test[j] = cv2.imread('S:\\Users\\Jkara\\OneDrive\\Documents\\CPEG_586\\Assignments_Workspace\\CPEG_586_Assignment3\\Data\\Test10000\\{0}'.format(filename),0)/255.0 
    j += 1

trainX = train.reshape(train.shape[0],train.shape[1]*train.shape[2])
testX = test.reshape(test.shape[0],test.shape[1]*test.shape[2])

print("End of Setup.\n")

HiddenLayer1Neurons = 100
OutputLayerNeurons = 10
numEpochs = 150

w1 = np.random.uniform(low=-0.1,high=0.1,size=(HiddenLayer1Neurons,784))
b1 = np.random.uniform(low=-1.0,high=1.0,size=(HiddenLayer1Neurons,1))
w2 = np.random.uniform(low=-0.1,high=0.1,size=(OutputLayerNeurons,HiddenLayer1Neurons))
b2 = np.random.uniform(low=-1.0,high=1.0,size=(OutputLayerNeurons,1))
learningRate = 0.1

def sigmoid(s):
    # res = np.empty((len(s),1),dtype='float64')
    # for x in range(0,len(s)):
    #     exponent = -1.0*s[x,0]
    #     res[x,0] = (1.0/(1.0 + math.e**exponent)) #Sigmoid
    return 1/(1+np.exp(-1*s))

def logLoss(a,y):
    if len(a) != len(y):
        raise(IndexError("logLoss: The size of a and y have to be the same."))
    resLoss = 0
    for z in range(0,len(a)):
        resLoss += -(y[z,0] * np.log(a[z,0]) + (1.0 - y[z,0]) * np.log(1.0 - a[z,0]))
    return resLoss

for ep in range(0,numEpochs):
    # Shuffle data between before each Epoch.
    trainX,trainY = shuffle(trainX, trainY)
    loss = 0
    
    for i in range(0,trainingImages):
        #Forward Pass
        currImg = trainX[i]
        currImg = currImg.reshape(len(currImg),1)
        #print(str(currImg))
        s1 = np.dot(w1,currImg) + b1
        a1 = sigmoid(s1)
        s2 = np.dot(w2,a1) + b2
        a2 = sigmoid(s2)
        
        #loss = logLoss(a2,trainY[i])
        loss += (0.5*((a2 - trainY[i])*(a2 - trainY[i]))).sum()
        # if i<100:
        #     print("Loss: " + str(loss))


        # Back Propagation
        #delta2 = ((trainY[i] - a2).T).dot(a2.dot((1.0 - a2).T))
        #delta1 = (delta2.dot(w2)).dot(a1.dot((1.0 - a1).T))
        yma2 = trainY[i] - a2 # y minus a2
        aoma2 = a2*(1.0 - a2) #a2 * one minus a2
        delta2 = ((yma2) * (aoma2)).T
        aoma = a1*(1.0 - a1) # a1 * one minus a1
        delta1 = (np.dot(delta2,w2)).T * aoma # (((delta2).dot(w2)).T * (aoma)) 

        # gradW2 = (delta2.T).dot(a1.T)
        gradW2 = np.dot(a1, delta2).T #(delta2.T).dot(a1.T)
        gradB2 = delta2.T
        gradW1 = np.dot(delta1,currImg.T) #(delta1).dot(currImg.T)
        gradB1 = delta1

        #update weights and biases
        w1 = w1 - learningRate * -1.0*gradW1
        w2 = w2 - learningRate * -1.0*gradW2
        b1 = b1 - learningRate * -1.0*gradB1
        b2 = b2 - learningRate * -1.0*gradB2

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
    a1 = 1/(1+np.exp(-1*s1)) # np.exp operates on the array
    s2 = np.dot(w2,a1) + b2
    a2 = 1/(1+np.exp(-1*s2))
    # determine index of maximum output value
    a2index = a2.argmax(axis = 0)
    currY = testY[i]
    # print(str(currY))
    # if (testY[i,a2index,0] == 1):
    if (currY[a2index,0] == 1):
        accuracyCount = accuracyCount + 1
print("Accuracy count = " + str(accuracyCount/testImages))