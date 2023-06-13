#import sys; args = sys.argv[1:]
import time
from collections import deque
from heapq import heappush, heappop, heapify
import numpy as np
import math
import random
import tensorflow as tf
from tensorflow import keras
line_list = []
with open("slidedata.txt") as f:
    count = 1
    for line in f:
        if count % 2 == 0: 
            l = line.strip().split(" ")
            xLis = [int(l2) for l2 in l[0]]
            yLis = [int(l2) for l2 in l[1]]
            line_list.append((xLis, yLis))
        count += 1

def dotProdMat(x, y): return [x1*y1 for (x1,y1) in zip(x,y)]
def dotProd(x, y): return sum(dotProdMat(x, y))

def perceptron(A, x, w_list):
    xLen = len(x)
    temp = x
    out = []
    for w in w_list[:-1]:
        w2 = [w[y : y + xLen] for y in range(0, len(w), xLen)]
        out.append([A(dotProd(temp, w3)) for w3 in w2])
        temp = out[len(out) - 1]
        xLen = len(temp)
    if len(out) > 0: out2 = out[len(out) - 1]
    else: out2 = temp
    output = dotProdMat(out2, w_list[len(w_list) - 1])
    out.append(output)
    return out

def sigmoid(num): return 1/(1+math.exp(-num))

def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))

def relu(x):
    if x > 0: return x
    else: return 0
def reluprime(x):
    if x > 0: return 1
    else: return 0

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

def p_net(A, x, w_list, b_list):
    new_a = np.vectorize(A)
    a = list()
    a.append(np.array(list(x)))
    for l in range(1, len(w_list) + 1): a.append(new_a((a[l-1]@w_list[l-1]) + b_list[l-1]))
    return a[len(w_list)]

def backPropWithChecker(epochs, f, fP, trainingSet, w, b, lam):
    AFunct = np.vectorize(f)
    APrime = np.vectorize(fP)
    for e in range(epochs):
        print(e)
        for z in trainingSet:
            #print(z)
            dot = [None]
            x, y  = z
            a = []
            a.append(x)
            for l in range(1, len(w)):
                dot.append((a[l-1]@w[l] + b[l]))
                a.append(AFunct(dot[l]))
            n = len(dot) - 1
            #for y2 in y:
            delta = [None for i in range(0, n)]
            #for y2 in y: delta.append((APrime(dot[n]))*(y2-a[n]))
            # if len(y) > 1: 
            #     y3 = []
            #     for y2 in y: y3 += list(y2)
            #     y = np.array([y3])
            # print(y)
            #print(y)
            delta.append((APrime(dot[n]))*(np.array(y)-a[n]))
            for l in range(n - 1, 0, -1): delta[l] = (APrime(dot[l]))*(delta[l+1]@(w[l+1].transpose()))
            for l in range(1, len(w)):
                b[l] = b[l] + (lam*delta[l])
                #print(a[l-1])
                # if len(a[l-1].transpose()) == len(delta[l]): w[l] = w[l] + (lam*((a[l-1].transpose())@delta[l]))
                # else: w[l] = w[l] + (lam*((a[l-1].transpose())*delta[l]))
                #print(delta[l])
                w[l] = w[l] + (lam*((a[l-1].transpose())*delta[l]))
    return [w[1:], b[1:]]

# print(len(trainingSet[0][0][0]))
trainingSet = line_list
x_train = []
y_train = []
for t in trainingSet:
    x, y = t
    x_train.append(x)
    y_train.append(y)
x_train = np.array(x_train)
y_train = np.array(y_train)
model = tf.keras.models.Sequential([
    #tf.keras.layers.Flatten(input_shape=(1, 64)),
    tf.keras.layers.Dense(64, activation = 'relu', use_bias=False),
    tf.keras.layers.Dense(32, activation = 'relu', use_bias=False),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(16, activation = 'relu', use_bias=False),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(8, activation = 'relu', use_bias=False),
    #tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(4, activation = 'softmax', use_bias=False)
])
#opt = keras.optimizers.Adam(learning_rate=0.01)
opt = 'adam'
model.compile(loss = 'mse', optimizer=opt, metrics = ['accuracy'])
model.fit(x_train, y_train, epochs = 1)
# predictions = model.predict(x_train)
# count = 0
# count2 = 0
# for p in predictions: 
#     print(p)
#     print()
#     max = p[0]
#     for p2 in p: 
#         if p2 > max: max = p2
#     x, y = trainingSet[count]
#     if y[list(p).index(max)] == 1: count2 += 1
#     count += 1
# print(count2/len(predictions))
# with open('predictions.txt', 'w+'):
#     for p in predictions: 
#         f.write(str(list(p)))
#         f.write('/n')
with open('nnsliding7.txt', 'w+') as f:
    for x in range(1, len(model.layers)):
        for b2 in list(model.layers[x].weights):
            b2 = b2.numpy().tolist()
            if type(b2[0]) is list:
                b3 = []
                for b in b2:
                    b4 = []
                    for b5 in b: b4.append(round(b5, 3))
                    b3.append(b4)
            else:
                b3 = []
                for b in b2: b3.append(round(b, 3))
            # print(str(b2).index(']]'))
            # print(str(b2)[str(b2).index(']]') + 2])
            #if x == 1: print(list(str(b2)[0: str(b2).index(']]')]))
            f.write(str(b3))
            f.write('\n')
            #for b3 in b2:
                # print()
                # print((list(b2)))
                # if count < len(list(b2)) - 1: f.write(str(list(b3)) + ",")
                # else: f.write(str(list(b3)))
                # f.write('\n')
                # count += 1

             # count += 1
            # b3 = []
            # count = 0
            # for b4 in b2:
            #     count += 1
            #     if count < len(b2):
            #         b5 = []
            #         for b6 in b4: b5.append(float(round(b6, 2)))
            #         b3.append(b5)
            #     else: 
            #         b5 = []
            #         print(b4)
            # f.write(str(b3))
            # f.write('\n')
        f.write('\n')
        f.write('\n')
        # for b2 in model.layers[x].bias.numpy():
        #     count = 0
        #     for b3 in b2:
        #         if count < len(b2) - 1: f.write(str(list(b3)) + ",")
        #         else: f.write(str(list(b3)))
        #         f.write('\n')
        #         count += 1
        # f.write('\n')
        # f.write('/n')

#biases = np.array([b[0] for b in w[1]])
#print(biases)
# count = 0
# count2 = 0
# for z in trainingSet:
#     x, y = z
#     p = p_net(relu, x, weights, biases)
#     print(p)
#     print(x)
#     print()
#     max2 = p[0][0]
#     c = 0
#     c2 = 0
#     for p2 in p[0]:
#         if p2 > max2: 
#             max2 = p2
#             c = c2
#         c2 += 1
#     if list(y[0]).index(1) == c: count2 += 1
#     # print(p[len(directionList[count])-1])
#     # print(y)
#     # if count == 100: 
#     #     # print(p[:len(y)])
#     #     # print(y)
#     #     break
#     count += 1
# print(count2/count)
#print((trainingSet))
#print(trainingSet[0])
#Arav Singh, pd. 2, 2023