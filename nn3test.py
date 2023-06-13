import sys; args = sys.argv[1:]
myFile = open(args[0]).read().splitlines()
import math

def makeDerived(sqrWeights, layers, output, operator):

    combined = []
    for i in range(len(layers)-1):
        if i == 0:
            tmp = []

            #handle the x
            counter = 0
            for x in range(layers[1]):
                #print(counter)
                tmp.append(sqrWeights[0][counter]/output)
                counter+=1
                tmp.append(0)
                tmp.append(sqrWeights[0][counter])
                counter+=1

            #handle the y
            counter = 0
            for x in range(layers[1]):
                tmp.append(0)
                tmp.append(sqrWeights[0][counter]/output)
                tmp.append(sqrWeights[0][counter+1])
                counter+=2
            #tmp = [num/output for num in tmp]
            combined.append(tmp)

        elif i == len(layers)-2:
            tmp = [sqrWeights[-1][-1], sqrWeights[-1][-1]]
            combined.append(tmp)

        else:
            tmp = []

            #handle the x
            counter = 0
            for a in range(layers[i+1]):
                for b in range(layers[i]):
                    tmp.append(sqrWeights[i][counter])
                    counter+=1
                for b in range(layers[i]):
                    tmp.append(0)

            #handle the y
            counter = 0
            for a in range(layers[i+1]):
                for b in range(layers[i]):
                    tmp.append(0)
                for b in range(layers[i]):
                    tmp.append(sqrWeights[i][counter])
                    counter+=1
            combined.append(tmp)
    
    if operator == ">": combined.append([(1+math.e)/(2*math.e)])
    else: 
        combined.append([(1+math.e)/(2)])
        for i in range(len(combined[-2])): combined[-2][i]*=-1

    return combined


def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def main():
    sqrWeights = []
    for line in myFile:
        if line and has_numbers(line):
            tmpLst = line.split(" ")
            newLst = []
            for num in tmpLst:
                num = num.replace(",", "")
                if num and has_numbers(num): 
                    newLst.append(float(num))
            sqrWeights.append(newLst)
    output, out = "", False
    operator = "<"
    for char in args[1]:
        if char == ">" or char == "<": 
            operator = char
            out = True
        elif out and char != "=": 
            output+=char

    output = float(output)
    layers = [2]
    for sub in sqrWeights: layers.append(int(len(sub)/layers[-1]))

    network = makeDerived(sqrWeights, layers, float(math.sqrt(output)), operator)

    newLayers = [3]
    for sub in network: newLayers.append(int(len(sub)/newLayers[-1]))

    #for thing in network: print(thing)
    print("Layer Counts: ", end = "")
    for i in range(len(newLayers)): 
        if i == len(newLayers) - 1: print(str(newLayers[i]))
        else: print(str(newLayers[i]) + " ", end = "")
    for group in network:
        for weight in group:
            print(weight, end = " ")
        print()


main()

# Arnav Kadam, pd. 6, 2023