import sys
import ast
import math
import random
import numpy as np
import matplotlib.pyplot as plt
#let = sys.argv[1]

def intToBinary(num):
    b = str(bin(num))
    return int(b[2:])

def sigmoid(num): return 1/(1+math.exp(-num))

def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))

def relu(x):
    if x > 0: return x
    else: return 0
def reluprime(x):
    if x > 0: return 1
    else: return 0

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
    maxPerc = 0
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
        #if e % 3000 == 0 and e > 0: lam /= 1.1
        count = 0
        for z in trainingSet:
            #print(z)
            x, y = z
            p = p_net(f, x, w[1:], b[1:])
            p2 = p[0][0][0]
            # if p2 < 0.5: p2 = 0
            # else: p2 = 1
            #print(p2, y[0][0])
            print(p2, y[0][0])
            print(p2, y[0][0])
            if p2 == y[0][0]: count += 1
        # print("Epoch", e)
        # print(count/len(trainingSet))
        if count/len(trainingSet) > maxPerc: 
            maxPerc = count/len(trainingSet)
            maxWeight = w[1:]
            maxBias = b[1:]
        #if maxPerc > 0.95: break
        #print()
    return [w[1:], b[1:]]
layerCounts = [3, 4, 10, 1, 1]
trainingSet = []
k = -1.5
for x in range(0, 50): 
    k = random.uniform(-1.5, 1.5)
    trainingSet.append((np.array([[k, 1]]), np.array([[k**2]])))
    #k = k + 0.1
# for x in range(0, 10000):
#     k = random.randint(0, 1)
#     #print(k)
#     if k == 0:
#         # x1 = random.uniform(0, 1)
#         # y1 = random.uniform(0, 1)
#         # while y1 == x1**2: y1 = random.uniform(0, 1)
#         # blist1 = [x1, y1]
#         # expected = 0
#         # mainBList = [None, np.array([blist1]), np.array([[1]])]
#         # trainingSet.append((np.array([[blist1[0], blist1[1]]]), np.array([[expected]])))
#         blist1 = [random.uniform(0, 1), random.uniform(0, 1)]
#         if (blist1[0]**2) == (blist1[1]): expected = 1
#         else: expected = 0
#         mainBList = [None, np.array([blist1]), np.array([[1]])]
#         trainingSet.append((np.array([[blist1[0], blist1[1]]]), np.array([[expected]])))
#     else:
#         x2 = random.uniform(0, 1)
#         blist1 = [x2, x2**2]
#         mainBList = [None, np.array([blist1]), np.array([[1]])]
#         expected = 1
#         trainingSet.append((np.array([[blist1[0], blist1[1]]]), np.array([[expected]])))
mainWList = [np.array([[random.uniform(0, 1) for z in range(0, layerCounts[y] * layerCounts[y + 1])]]) for y in range(0, len(layerCounts[:-1]))]
# mainWList = [None] + mainWList
# print(mainWList)
wlist1 = np.array([[random.uniform(-0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)], [random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]])
wlist2 = np.array([[random.uniform(0, 1)], [random.uniform(0, 1)], [random.uniform(0, 1)], [random.uniform(0, 1)]])
mainWList = [None, wlist1, wlist2]
blist1 = np.array([[random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)]])
blist2 = np.array([[random.uniform(0, 1)]])
mainBList = [None, blist1, blist2]
# back = backPropWithChecker(10000, relu, reluprime, trainingSet, mainWList, mainBList, 0.0999999)
# print(back)

weights = [np.array([[ 6.93108673e-10, -3.30687899e+00,  1.66140544e+00,
         1.75449455e+00],
       [-4.40167095e-01, -6.95553251e-01, -3.64193727e-01,
        -4.30956461e-01]]), np.array([[9.39351083e-12],
       [6.72018386e-01],
       [5.53132932e-01],
       [5.83774252e-01]])]
biases = [np.array([[ 0.4401671 , -0.92965955, -0.41692607, -0.39455762]]), np.array([[0.07617293]])]
nodes = [2, 4, 3, 1, 1]
k = -1.5
realX = []
realY = []
fakeY = []
intersections = []
for x in range(0, 31):
    inputs = [k, 1]
    ecp = k**2
    xVal = k
    realVal = ecp
    #print(p_net(relu, inputs, weights, biases))
    fakeVal = p_net(relu, inputs, weights, biases)[0][0]
    realX.append(xVal)
    realY.append(ecp)
    fakeY.append(fakeVal)
    print(fakeVal, ecp)
    if ecp == fakeVal: intersections.append((xVal, ecp))
    k += 0.1
intersections.append((-0.649, 0.421))
intersections.append((-0.279, 0.078))
intersections.append((0.279, 0.078))
intersections.append((0.649, 0.421))
intersections.append((1.292, 1.669))
breakpoints = []
breakpoints.append((-0.502, 0.1))
breakpoints.append((-0.407, 0.078))
breakpoints.append((0.399, 0.078))
breakpoints.append((0.502, 0.137))
print(intersections)
plt.axis([-1.5, 1.5, 0, 2.25])
plt.plot(realX, realY)
plt.plot(realX, fakeY, color="red")
plt.show()