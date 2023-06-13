import math, random
import matplotlib.pyplot as plt
def backPropogatePartials(intermediary, output, errors):
    partials = []
    for i in range(len(intermediary)-1, 0, -1):
        if i == len(intermediary)-1:
            tmpList = []
            for x in range(len(intermediary[-1])): tmpList.append((output - intermediary[-1][x]) * intermediary[-2][x])
            partials.append(tmpList)
        else:
            tmpPartials = []
            for x in range(len(intermediary[i])):
                for j in range(len(intermediary[i-1])):
                    val = intermediary[i-1][j] * errors[i-1][x]
                    tmpPartials.append(val)
            partials.append(tmpPartials)
    return partials[::-1]

def backPropogateErrors(intermediary, output, weights):
    errors = []
    for i in range(len(intermediary)-1, 0, -1):
        if i == len(intermediary)-1:
            tmpList = []
            for x in range(len(intermediary[-1])): tmpList.append(output - intermediary[-1][x])
            errors.append(tmpList)
        elif i == len(intermediary) - 2:
            tmpList = []
            for x in range(len(intermediary[-2])): tmpList.append((output - intermediary[-1][x]) * weights[-1][x] * derivative(intermediary[-2][x]))
            errors.append(tmpList)
        else:
            tmpErrors = []
            weightCounter = 0
            for x in range(len(intermediary[i])):
                cumError = 0
                for j in range(len(intermediary[i+1])):
                    tmpError = (weights[i][weightCounter] * errors[-1][j]) * (derivative(intermediary[i][x]))
                    cumError+=tmpError
                    weightCounter+=1
                tmpErrors.append(cumError)
            errors.append(tmpErrors)
    return errors[::-1]

def forwardPropogate(inputs, weights, nodes, ecp):
    currLayer = inputs
    intermediary = [inputs]
    for i in range(0, len(weights)-1):
        currWeights = weights[i]
        numNodes = nodes[i]
        tempLayer = []
        weightLooper = 0
        for k in range(0, numNodes):
            newNum = 0
            for x in range(len(currLayer)):
                newNum+=currWeights[weightLooper] * currLayer[x]
                weightLooper+=1
            tempLayer.append(transfer(newNum))
        currLayer = tempLayer
        intermediary.append(currLayer)
    outputs = []
    for i in range(len(currLayer)):
        outputs.append(currLayer[i] * weights[len(weights)-1][i])
    intermediary.append(outputs)
    print(str(inputs[0]) + " " + str(ecp) + " " + str(intermediary[-1]))
    return intermediary

def transfer(x): return x if x > 0 else 0
def derivative(x): 
    if x <= 0: return 0
    else: return 1

def updateWeights(weights, partials, rate):
    for i in range(len(partials)):
        for x in range(len(partials[i])): weights[i][x] += partials[i][x] * rate
    return weights

def main():
    nodes = [2, 4, 3, 1, 1]
    #numWeights = [12, nodes[1]*nodes[0], nodes[1], 1]
    #numWeights = [8, 32, 8, 1, 1]
    numWeights = [4, 8, 12, 3, 1]

    weights = []
    for val in numWeights: weights.append([random.uniform(0, 1) for x in range(val)])

    trainers = []

    #make training data
    for i in range(50):
        x = random.uniform(-1.5, 1.5)
        y=x**2
        trainers.append([[x, 1], y])

    a = 0.1
    for i in range(10000): #start propogating
        for inputs, ecp in trainers:
            #plt.plot(inputs[0], ecp, "r.")
            intermediary = forwardPropogate(inputs, weights, nodes, ecp)
            #print(intermediary)
            errors = backPropogateErrors(intermediary, ecp, weights)
            #print()
            #print(errors)
            partials = backPropogatePartials(intermediary, ecp, errors)
            #print(partials)
            weights = updateWeights(weights, partials, a)
            #plt.plot(input)
        if i % 3000 == 0 and i > 0: a /= 1.1
    print("Layer Counts: 3 4 8 1 1")
    print(weights)
    # for group in weights:
    #     for weight in group:
    #         print(weight, end = " ")
    #     print()

#main()
#weights = [[-1.3439479294463772, -1.3401052748852202, -1.3478116573790684, -0.3783532921510304], [0.42413054791926563, 0.5606094583617285, 0.7467970539635107, 0.7332615078954738, 0.5294920055623943, 0.4668703912347484, 0.429696206409137, -9.198963432074288e-06], [0.42906266207003124, 0.0932957012737435, 0.6708108902644339, 0.9234112118584843, 0.7700137878240878, 0.4229417837407307, 0.7353399967799706, 0.08830013025712133, 0.18329102648233322, 0.08358358507052994, 0.8551882995972823, 0.6747798919453804, 0.0867903712395573, 0.33190476018580956, 0.26598155551300806, 0.01806429079164959, -0.12571947423917187, 0.6813649288859687, 0.5885357820033462, 0.9593346890638018, 0.7709705438666767, 0.19778506460046555, 0.15320220381070385, 0.3259074201490325, 0.3807402031846319, -0.03177944364850713, 0.6180649180686528, 0.8215820254616709, -0.03880555296286801, 0.5973501989711159, 0.6936669998241651, 0.9979362082203321], [-0.0980290958745158, 0.6537987903534422, 0.32769344115201327, 0.3302802025362599, 0.02136425838860093, 0.6146169854772469, 0.3840628189259788, -0.18434113535031155], [0.6488468459769158]]
weights = [[3.419465887062523, -1.478093492326395, -1.4327111143739093, -0.5356224183098326], [-0.00023167474358423267, -0.000672334447621898, 0.8481859145451369, -1.2409924102690754e-05, 2.489353593325678, 3.733723657450071, -0.0007357229969637696, -8.433091249781133e-05], [0.10332343956306352, -0.47857643355715784, 0.6151679623324308, 0.18561661825205245, 1.1647581947841392, -0.05830474674573542, 0.07495108281325057, 0.6079408646754042, 1.7073843374819888, -0.3107812351299208, 0.399489695215284, 0.25542690685572], [0.5184235751982512, 0.06316459919726787, 0.33666493821380405], [0.6213662772744623]]
nodes = [2, 4, 3, 1, 1]
k = -1.5
realX = []
realY = []
fakeY = []
for x in range(0, 31):
    inputs = [k, 1]
    ecp = k**2
    xVal = k
    realVal = ecp
    fakeVal = forwardPropogate(inputs, weights, nodes, ecp)[-1]
    realX.append(xVal)
    realY.append(ecp)
    fakeY.append(fakeVal)
    k += 0.1
plt.axis([-1.5, 1.5, 0, 2.25])
plt.plot(realX, realY)
plt.plot(realX, fakeY, color="red")
plt.show()