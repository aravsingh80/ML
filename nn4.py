import sys; args = sys.argv[1:]
myFile = open(args[0]).read().splitlines()
import math
import random
inputs = args[1]
def findFactors(num):
    factors = []
    x = 1
    while x <= math.sqrt(num):      
        if (num % x == 0) :
            if (num / x == x): factors.append(x)
            else: 
                factors.append(x)
                factors.append(int(num/x))
        x += 1
    return factors
def validLine(s):
    for x in s:
        if x.isdigit(): return True
    return False
def anyDigits(s):
    if len(s) == 0: return False
    for x in s:
        if x != "-" and x != "." and not x.isdigit(): return False
    return True
weights = [[x for x in myFile[y].split(" ")] for y in range(len(myFile)) if validLine(myFile[y])]
mainWList = []
for x in weights:
    w = []
    for y in x:
        if ',' in y: y = y[0:len(y)-1]
        if anyDigits(y): w.append(float(y))
    mainWList.append(w)
layerCountLen = len(mainWList) + 1
initlayerCounts = [len(x) for x in mainWList]
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
layerCounts = [2]
for x in mainWList: layerCounts.append(int(len(x)/layerCounts[len(layerCounts)-1]))
#layerCounts += [int(len(x)/layerCounts[len(layerCounts)-1]) for x in mainWList]
neurNet = []
for x in range(0, len(layerCounts) - 1):
    if x == 0:
        xY = []
        ind = 0
        for y in range(0, layerCounts[1]):
            adder = mainWList[0][ind] / float(math.sqrt(radius))
            xY.append(adder)
            ind += 1
            xY.append(0)
            xY.append(mainWList[0][ind])
            ind += 1
        ind = 0
        for y in range(0, layerCounts[1]):
            xY.append(0)
            adder = mainWList[0][ind] / float(math.sqrt(radius))
            xY.append(adder)
            xY.append(mainWList[0][ind+1])
            ind += 2
        neurNet.append(xY)

    elif x == len(layerCounts) - 2:
        finalW = mainWList[len(mainWList) - 1]
        xY = [finalW[len(finalW) - 1], finalW[len(finalW) - 1]]
        neurNet.append(xY)

    else:
        xY = []
        ind = 0
        for y in range(0, layerCounts[x + 1]):
            for z in range(0, layerCounts[x]):
                xY.append(mainWList[x][ind])
                ind += 1
            for z in range(0, layerCounts[x]):
                xY.append(0)
        ind = 0
        for y in range(0, layerCounts[x + 1]):
            for z in range(0, layerCounts[x]):
                xY.append(0)
            for z in range(0, layerCounts[x]):
                xY.append(mainWList[x][ind])
                ind+=1
        neurNet.append(xY)

if gr or grE: neurNet.append([(1 + math.e)/(2 * math.e)])
else: 
    neurNet.append([(1 + math.e)/2])
    for x in range(len(neurNet[len(neurNet) - 2])): neurNet[len(neurNet) - 2][x] *= -1
finalLayerCounts = [3]
for x in neurNet: finalLayerCounts.append(int(len(x)/finalLayerCounts[len(finalLayerCounts)-1]))
print("Layer counts: ", end = "")
for x in range(0, len(finalLayerCounts)): 
    if x == len(finalLayerCounts) - 1: print(str(finalLayerCounts[x]))
    else: print(str(finalLayerCounts[x]) + " ", end = "")
for x in neurNet:
    for y in x:
        print(y, end = " ")
    print()
#finalLayerCounts += [int(len(x)/layerCounts[len(finalLayerCounts)-1]) for x in neurNet]
# print("Layer counts:", finalLayerCounts)
# #if everything goes wrong, change to [3, 4, 10, 1, 1]
# print("Weights:")
# for w in mainWList: print(w)
#Arav Singh, pd. 2, 2023
# layerCounts = [1 for x in range(0, layerCountLen)]
# factorsList = [findFactors(x) for x in initlayerCounts]
# commonFactors = []
# for x in range(0, len(initlayerCounts) - 3): commonFactors.append([y for y in factorsList[x] if y in factorsList[x+1]])
# print(commonFactors)
# exit()
# layerCounts = [3] + layerCounts + [1, 1]
# print("Layer counts:", layerCounts)
# #if everything goes wrong, change to [3, 4, 10, 1, 1]
# print("Weights:")
# for w in mainWList: print(w)
# print("All inputs", inputs)
#Arav Singh, pd. 2, 2023