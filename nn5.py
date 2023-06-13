#import sys; args = sys.argv[1:]
import math
import random
#inputs = args[0]
def error(x, y): return [y[0] - x1 for x1 in x]
def error2(perc, expected, w, prime): 
    allErrors = []
    for x in range(len(perc) - 1, 0, -1):
        if x == len(perc) - 1:
            errors = []
            for x in range(len(perc[len(perc) - 1])): errors.append(expected - perc[len(perc) - 1][x])
            allErrors.append(errors)
        elif x == len(perc) - 2:
            errors = []
            for x in range(len(perc[len(perc) - 2])): errors.append((expected - perc[len(perc) - 1][x]) * w[-1][x] * prime(perc[len(perc) - 2][x]))
            allErrors.append(errors)
        else:
            errors = []
            count = 0
            for y in range(len(perc[x])):
                totalError = 0
                for z in range(len(perc[x + 1])):
                    error = (w[x][count] * allErrors[len(allErrors) - 1][z]) * (prime(perc[x][y]))
                    totalError += error
                    count += 1
                errors.append(totalError)
            allErrors.append(errors)
    return allErrors[::-1]
def dotProdMat(x, y): return [x1*y1 for (x1,y1) in zip(x,y)]
def dotProd(x, y): return sum(dotProdMat(x, y))
def sigmoid(num): 
    if num > 0: return 1/(1+math.exp(-num))
    else: return math.exp(num)/(1+math.exp(num))
def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))
def relu(x):
    if x > 0: return x
    else: return 0
def reluprime(x):
    if x > 0: return 1
    else: return 0
def intToBinary(num):
    b = str(bin(num))
    return int(b[2:])
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
def backProp(i, perc, w, a, expected, prime):
    #out = [error(perc[len(perc) - 1],expected)]
    out = error2(perc, expected[0], w, prime)
    begError = [out[len(out) - 1][0], out[len(out) - 2][0]]
    percOut = [i] + perc[:-1]
    percFinal = percOut[len(percOut) - 1]
    percOut = percOut[0 : len(percOut) - 1]
    gradient = [dotProdMat(out[len(out) - 1], percFinal)]
    complement = [1 - p for p in percFinal]
    out.append(dotProdMat(dotProdMat(dotProdMat(percFinal, complement), w[len(w) - 1]), out[len(out) - 1]))
    percWeight = [[percOut[x], w[:-1][x]] for x in range(0, len(percOut))]
    for x in percWeight[::-1]:
        per = x[0]
        perW = x[1]
        outFin = out[len(out) - 1]
        perW2 = [perW[y : y + len(per)] for y in range(0, len(perW), len(per))]
        outFinZero = [0 for x in range(0, len(per))]
        # if len(perW2)< len(outFin): l = len(perW2)
        # else: l = len(outFin)
        perWFin = [[perW2[y], outFin[y]] for y in range(0, len(perW2))]
        for y in perWFin:
            perW2y = y[0]
            outFiny = y[1]
            dotP = dotProdMat(perW2y, [outFiny] * len(per))
            outFinZero = dotProdMat(dotProdMat(per, [1 - p for p in per]), [outFinZero[y2] + dotP[y2] for y2 in range(0, len(dotP))])
        out.append(outFinZero)
        l = []
        for p in per:
            for y in range(0, len(outFin)): l.append(p)
        tempG = [dotProdMat(l, outFin * len(per))]
        for g in gradient: tempG.append(g)
        gradient = tempG
    toReturn = []
    wGrad = [[w[y], gradient[y]] for y in range(0, len(w))]
    for x in wGrad:
        w2 = x[0]
        grad2 = x[1]
        dotP2 = dotProdMat(grad2,[a] * len(grad2))
        if len(dotP2) < len(w2): 
            mainList = w2
            l = len(dotP2)
        else: 
            mainList = dotP2
            l = len(w2)
        toReturn.append([dotP2[y] + w2[y] for y in range(0, l)])
    return toReturn
epochs = 0
#a = 0.0999999
a = 0.1
outputs = []
currError = 1
layerCounts = [3, 4, 10, 1, 1]
#if everything goes wrong, change to [3, 4, 10, 1, 1]
errors = []
mainWList = [[random.uniform(-1, 1) for z in range(0, layerCounts[y] * layerCounts[y + 1])] for y in range(0, len(layerCounts[:-1]))]
trainingSet = []
for x in range(0, 800):
    if x < 400:
        blist1 = [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]
        mainBList = blist1 + [1]
        if (blist1[0]**2) == (blist1[1]): expected = 1
        else: expected = 0
        trainingSet.append([mainBList, [expected]])
    else:
        x2 = random.uniform(-1.5, 1.5)
        blist1 = [x2, x2**2]
        mainBList = blist1 + [1]
        expected = 1
        trainingSet.append([mainBList, [expected]])
count = 0
errorCounter = []
while epochs < 600:
    totalError = 1
    errorCounter = []
    for b, exp in trainingSet:
        output = perceptron(relu, b, mainWList)
        mainWList = backProp(mainBList, output, mainWList, a, exp, reluprime)
        errorCounter.append([b[0], b[1], perceptron(relu, b, mainWList,)[-1][0]])
        #totalError += (0.5 * (radius - endOutput)**2)
        # totalError += sum(currError)
        #if totalError < 0.5: a = .3
        #if totalError < 0.1: a += math.sqrt(abs(totalError))
        #if totalError < 0.05: a *= math.sqrt(abs(totalError))
        # if totalError < 0.00000005: a = math.sqrt(abs(totalError))
    c = 0
    for x in errorCounter: 
        if x[0]**2 != x[1] and x[2] < 0.5: c += 1
        elif x[0]**2 == x[1] and x[2] >= 0.5: c += 1
    print(c/len(errorCounter))
        #if count % 100 == 0: totalError = 1
        #     a *= sum(totalError)
        #     totalError.clear()
        # print("Layer counts:", [3, 4, 10, 1, 1])
        # #if everything goes wrong, change to [3, 4, 10, 1, 1]
        # print("Weights:")
        # for w in mainWList: print(w)
        # print("All inputs", inputs)
    # if totalError >= 0.5: a = 0.6
    # if totalError < 0.5: a = .3
    # if totalError < 0.1: a = 0.2
    # if totalError < 0.05: a = 0.1
    epochs += 1
    # print("Layer counts:", [3, 4, 10, 1, 1])
    # #if everything goes wrong, change to [3, 4, 10, 1, 1]
    # print("Weights:")
    # for w in mainWList: print(w)
    #print("All inputs", inputs)
#Arav Singh, pd. 2, 2023