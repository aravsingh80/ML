import sys; args = sys.argv[1:]
import math
import random
inputs = args[0]
gr = False
lt = False
grE = False
ltE = False
if '=' in inputs: 
    if '<' in inputs: ltE = True
    else: grE = True
    radius = float(inputs[inputs.index('=') + 1:])
elif '<' in inputs: 
    lt = True
    radius = float(inputs[inputs.index('<') + 1:])
else: 
    gr = True
    radius = float(inputs[inputs.index('>') + 1:])
def error(x, y): return [y[0] - x1 for x1 in x]
def dotProdMat(x, y): return [x1*y1 for (x1,y1) in zip(x,y)]
def dotProd(x, y): return sum(dotProdMat(x, y))
def totError(x): return [((1-x1)**2)/2 for x1 in x]
def sigmoid(num): 
    if num > 0: return 1/(1+math.exp(-num))
    else: return math.exp(num)/(1+math.exp(num))
def sigmoidPrime(num): return sigmoid(num) * (1 - sigmoid(num))
def intToBinary(num):
    b = str(bin(num))
    return int(b[2:])
def binToInt(s):
    totSum = 0
    for x in range(len(s) - 1, -1, -1):
        if s[x] == '1': totSum += (2**x)
    return totSum
def t(num, x):
    if "1" in num: return x
    elif "2" in num: 
        if x > 0: return x
        else: return 0
    elif "3" in num: return 1/(1+math.exp(-x))
    else: return (2/(1+math.exp(-x)))-1
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
def backProp(i, perc, w, a, expected):
    out = [error(perc[len(perc) - 1],expected)]
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
layerCounts = [3, 4, 8, 1, 1]
#if everything goes wrong, change to [3, 4, 10, 1, 1]
errors = []
mainWList = [[random.uniform(-1, 1) for z in range(0, layerCounts[y] * layerCounts[y + 1])] for y in range(0, len(layerCounts[:-1]))]
while epochs < 20000:
    # trainingSet = []
    # for z in range(0, 10000):
    #     x, y = random.uniform(-1, 1), random.uniform(-1, 1)
    #     val = math.sqrt((float(x)**2) + (float(y)**2))
    #     if val < 1: trainingSet.append([[[float(x), float(y)]], [[1]]])
    #     else: trainingSet.append([[[float(x), float(y)]], [[0]]])

    # wlist1 = [[random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)], [random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1), random.uniform(-1, 1)]]
    # wlist2 = [[random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)], [random.uniform(-1, 1)]]
    # mainWList = [wlist1, wlist2]
    blist1 = [random.uniform(-1.5, 1.5), random.uniform(-1.5, 1.5)]
    #blist2 = [[random.uniform(-1, 1)]]
    mainBList = blist1 + [1]
    if gr: 
        if (blist1[0]**2)+(blist1[1]**2) > radius: expected = 1
        else: expected = 0
    elif lt: 
        if (blist1[0]**2)+(blist1[1]**2) < radius: expected = 1
        else: expected = 0
    elif grE:
        if (blist1[0]**2)+(blist1[1]**2) >= radius: expected = 1
        else: expected = 0
    else: 
        if (blist1[0]**2)+(blist1[1]**2) <= radius: expected = 1
        else: expected = 0
    output = perceptron(sigmoid, mainBList, mainWList)
    mainWList = backProp(mainBList, output, mainWList, a, [expected])
    p = perceptron(sigmoid, mainBList, mainWList)
    endOutput = p[len(p) - 1][0]
    currError = (0.5 * (radius - endOutput)**2)
    a = math.log(currError)
    # if epochs % 250 == 0 and len(errors) > 0: 
    #     errors.pop()
    #     a = math.log(sum(errors))
    # if epochs % 2500 == 0: 
    #     #mainWList = [[random.uniform(-1, 1) for z in range(0, layerCounts[y] * layerCounts[y + 1])] for y in range(0, len(layerCounts[:-1]))]
    #     errors.clear()
    # errors.append(currError)
    #a = 0.1
    # if endOutput < 0.5: totOut = 0
    # else: totOut = 1
    epochs += 1
    # outputs.append(totOut)
    # print("Output:", totOut)
    print("Layer counts:", [3, 4, 8, 1, 1])
    #if everything goes wrong, change to [3, 4, 10, 1, 1]
    print("Weights:")
    for w in mainWList: print(w)
    print("All inputs", inputs)
#Arav Singh, pd. 2, 2023