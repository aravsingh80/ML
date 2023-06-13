import sys
import ast
import math
import random
import numpy as np
let = "C"

def intToBinary(num):
    b = str(bin(num))
    return int(b[2:])

def sigmoid(num): return 1/(1+math.exp(-num))

def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))

def p_net(A, x, w_list, b_list):
    new_a = np.vectorize(A)
    a = list()
    a.append(np.array([list(x)]))
    for l in range(1, len(w_list) + 1): a.append(new_a((a[l-1]@w_list[l-1]) + b_list[l-1]))
    return a[len(w_list)]

def backProp(epochs, f, fP, trainingSet, w, b, lam):
    dot = [None]
    AFunct = np.vectorize(f)
    APrime = np.vectorize(fP)
    for e in range(0, epochs):
        for z in trainingSet:
            x, y  = z
            a = []
            a.append(x)
            for l in range(1, len(w)):
                dot.append((a[l-1]@w[l] + b[l]))
                a.append(AFunct(dot[l]))
            delta = [None for n in range(0, len(dot) - 1)]
            n = len(dot) - 1
            delta.append((APrime(dot[n]))*(y-a[n]))
            for l in range(n - 1, 0, -1): delta[l] = (APrime(dot[l]))*(delta[l+1]@(w[l+1].transpose()))
            for l in range(1, len(w)):
                b[l] = b[l] + (lam*delta[l])
                w[l] = w[l] + (lam*((a[l-1].transpose())@delta[l]))
            dot = [None]
        print("Output Vector:", a[len(a) - 1], "for", x)
    return [w[1:], b[1:]]

def backPropWithChecker(epochs, f, fP, trainingSet, w, b, lam):
    dot = [None]
    for e in range(0, epochs):
        AFunct = np.vectorize(f)
        APrime = np.vectorize(fP)
        for z in trainingSet:
            x, y  = z
            a = []
            a.append(np.array([list(x)]))
            for l in range(1, len(w)):
                dot.append((a[l-1]@w[l] + b[l]))
                a.append(AFunct(dot[l]))
            delta = [None for n in range(0, len(dot) - 1)]
            n = len(dot) - 1
            delta.append(APrime(dot[n])*(y-a[n]))
            for l in range(n - 1, 0, -1): delta[l] = APrime(dot[l])*(delta[l+1]@(w[l+1].transpose()))
            for l in range(1, len(w)):
                b[l] = b[l] + (lam*delta[l][0])
                w[l] = w[l] + (lam*(a[l-1][0].transpose()@delta[l][0]))
            dot = [None]
        count = 0
        for z in trainingSet:
            x, y = z
            p = p_net(sigmoid, x, w[1:], b[1:])
            p2 = p[0][0][0]
            if p2 < 0.5: p2 = 0
            else: p2 = 1
            #print(p2, y[0][0])
            if p2 == y[0][0]: count += 1
        print("Epoch", e)
        print(count/len(trainingSet))
        print()
    return [w[1:], b[1:]]

if let == "s" or let == "S":
    def SUM(x, y): 
        b = str(intToBinary(x+y))
        while len(b) < 2: b = "0" + b
        return (int(b[len(b) - 2]), int(b[len(b) - 1]))
    trainingSet = [(np.array([[0, 0]]), np.array([[0, 0]])), (np.array([[0, 1]]), np.array([[0, 1]])), (np.array([[1, 0]]), np.array([[0, 1]])), (np.array([[1, 1]]), np.array([[1, 0]]))]
    wlist1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]]) 
    wlist2 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1)]])
    mainWList = [None, wlist1, wlist2]
    blist1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]])
    blist2 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1)]])
    mainBList = [None, blist1, blist2]
    count = 0
    back = backProp(12500, sigmoid, sigmoidPrime, trainingSet, mainWList, mainBList, 0.0999999) #5.295432542517969
    w = back[0]
    b = back[1]
    # p = p_net(sigmoid, x, w, b)
    # p2 = p[0][0][0]
    # p3 = p[0][0][1]
        # print(p2, p3)
        # if p2 < 0.5: p2 = 0
        # else: p2 = 1
        # if p3 < 0.5: p3 = 0
        # else: p3 = 1
        # x, y = tuple(z[1][0])
        # if p2 == x and p3 == y: count += 1
    print(count / len(trainingSet))
if let == "c" or let == "C":
    with open("10000_pairs.txt") as f: line_list = [tuple(line.strip().split(" ")) for line in f]
    trainingSet = []
    for z in line_list:
        x, y = z
        val = math.sqrt((float(x)**2) + (float(y)**2))
        if val < 1: trainingSet.append((np.array([[float(x), float(y)]]), np.array([[1]])))
        else: trainingSet.append((np.array([[float(x), float(y)]]), np.array([[0]])))
    wlist1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
    wlist2 = np.array([[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]])
    mainWList = [None, wlist1, wlist2]
    blist1 = np.array([[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]])
    blist2 = np.array([[random.uniform(-1, 1)]])
    mainBList = [None, blist1, blist2]
    back = backPropWithChecker(100, sigmoid, sigmoidPrime, trainingSet, mainWList, mainBList, 0.0999999)