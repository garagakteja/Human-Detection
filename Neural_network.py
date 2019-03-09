import numpy as np
import random
import cv2
import math
import hog_file #HOG Descriptor source code

rubric = [] #Contains whether a training image is human (=1) or not (=0)
sample_train=np.zeros((20,7524))

print("Training the Classifier..................................")
# Computing HOG Descriptors for given Training images
sample_train[0]=hog_file.hog_descriptor("trainpos1.bmp")
rubric.append(1)

sample_train[1]=hog_file.hog_descriptor("trainpos2.bmp")
rubric.append(1)

sample_train[2]=hog_file.hog_descriptor("trainpos3.bmp")
rubric.append(1)

sample_train[3]=hog_file.hog_descriptor("trainpos4.bmp")
rubric.append(1)

sample_train[4]=hog_file.hog_descriptor("trainpos5.bmp")
rubric.append(1)

sample_train[5]=hog_file.hog_descriptor("trainpos6.bmp")
rubric.append(1)

sample_train[6]=hog_file.hog_descriptor("trainpos7.bmp")
rubric.append(1)

sample_train[7]=hog_file.hog_descriptor("trainpos8.bmp")
rubric.append(1)

sample_train[8]=hog_file.hog_descriptor("trainpos9.bmp")
rubric.append(1)

sample_train[9]=hog_file.hog_descriptor("trainpos10.bmp")
rubric.append(1)

sample_train[10]=hog_file.hog_descriptor("trainneg1.bmp")
rubric.append(0)

sample_train[11]=hog_file.hog_descriptor("trainneg2.bmp")
rubric.append(0)

sample_train[12]=hog_file.hog_descriptor("trainneg3.bmp")
rubric.append(0)

sample_train[13]=hog_file.hog_descriptor("trainneg4.bmp")
rubric.append(0)

sample_train[14]=hog_file.hog_descriptor("trainneg5.bmp")
rubric.append(0)

sample_train[15]=hog_file.hog_descriptor("trainneg6.bmp")
rubric.append(0)

sample_train[16]=hog_file.hog_descriptor("trainneg7.bmp")
rubric.append(0)

sample_train[17]=hog_file.hog_descriptor("trainneg8.bmp")
rubric.append(0)

sample_train[18]=hog_file.hog_descriptor("trainneg9.bmp")
rubric.append(0)

sample_train[19]=hog_file.hog_descriptor("trainneg10.bmp")
rubric.append(0)



op = list(zip(sample_train,rubric))
random.shuffle(op)

M, N = zip(*op)

#Below Function is used for training the Neural Network
def genNeuralNetwork(M,N,hd_neurons_count): # hd_neurons_count is the desired no of hidden layer neurons

    np.random.seed(1)
    #Random values for weights of neurons
    w1 = np.random.randn(hd_neurons_count, len(M[0])) * 0.01
    b1 = np.zeros((hd_neurons_count,1))
    w2 = np.random.randn(1,hd_neurons_count) * 0.01
    b2 = np.zeros((1,1))
     # M = input_data of 20 images
     # N = Result of image, Positive = 1 and Negative = 0. Pos
    
    dictionary = {}
    
    variable=0
    cost = 0
    for i in range(0,200):
        
        cost = 0
        for j in range(0,len(M)):
            features = M[j].shape[0]
            q = M[j].reshape(1,features)
            q = q.T
            v1 = w1.dot(q)+ b1   #Multiplication for Level 1 hidden layer
            a1 = ReLu(v1)
            v2 = w2.dot(a1) + b2
            a2 = sigmoid(v2)
            
            # Backward Propogation Method Implementation
            diff2 = (a2-N[j])  *  Sigmoid_Derivative(a2)    #Difference calculation
            dw2 = np.dot(diff2,a1.T)
            db2 = np.sum(diff2,axis=1, keepdims=True)
            #ReLu 
            diff1 = w2.T.dot(diff2) * ReLuDerivation(a1)
            
            dw1 =  np.dot(diff1,q.T)
            db1 =  np.sum(diff1,axis=1, keepdims=True)

            #updating weights
            w1 = w1 - 0.01*dw1
            w2 = w2 - 0.01*dw2
            b1 = b1 - 0.01*db1
            b2 = b2 - 0.01*db2
            dictionary = {'w1':w1,'b1':b1,'w2':w2,'b2':b2}

            cost += (1.0/2.0)*((abs(a2-N[j]))**2)
        cost_avg = cost/20.0
        print("Epoch Number = ",i,"Cost Average = ",cost_avg)
        if i>1:
            if(abs(cost_avg - variable)<0.00001):
                break
        variable = cost_avg
        
    return dictionary




# Return value between 0 and 1.
def sigmoid(val):
    return 1.0 / (1.0 + np.exp(-val))

#Sigmoid Implementation
def Sigmoid_Derivative(x):
    return x * (1-x)

# For ForwardFeed, It return original value if it's >0 , or 0 if value
def ReLu(val):
    return val*(val>0)

# Backward Propogration with return value to be 1 or 0.
def ReLuDerivation(x):
    return 1. * (x > 0)

dictionary = genNeuralNetwork(M,N,250) #Enter the number of Hidden Layer Neurons as the third paramter here.

n = 1
picture_info = []
for n in range(1,11):
    #Reads test images
    name = "test1"+str(n)+".bmp"
    print(name)
    Vector1=hog_file.hog_descriptor(name)
    picture_info.append(Vector1)
new_picture_info = np.array(picture_info)



def prediction(M,dictionary):  #Function called to check if test image into Human or Not.
    w1 = dictionary['w1']
    w2 = dictionary['w2']
    b2 = dictionary['b2']
    b1 = dictionary['b1']
    features = M.shape[0]
    q = M.reshape(1,features)
    v1 = np.dot(w1,q.T) + b1   #Multiplication for Level 1 hidden layer
    a1 = ReLu(v1)
    v2 = np.dot(w2,a1) + b2
    a2 = sigmoid(v2)
    return a2

for image_vctr in new_picture_info:
    if(prediction(image_vctr,dictionary)>0.5): #Classification Decision.
        print("Human has been Detected in this image.")
    else:
        print("Human has NOT been Detected in this image.")
    print(prediction(image_vctr,dictionary)) #Computed Value at the Output neuron for each test image