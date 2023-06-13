import sys; args = sys.argv[1:]
myFile = open(args[0], "r").read().splitlines()
import math
a = args[1]
i = [float(args[x]) for x in range(2, len(args))]
w = [[float(x) for x in myFile[y].split(" ")] for y in range(len(myFile))]
def t(num, x):
    if "1" in num: return x
    elif "2" in num: 
        if x > 0: return x
        else: return 0
    elif "3" in num: return 1/(1+math.exp(-x))
    else: return (2/(1+math.exp(-x)))-1
def perceptron(A, x, w_list):
    curr = x
    for l in range(0, len(w_list) - 1): 
        currW = w_list[l]
        temp = []
        count2 = int(len(currW)/len(curr))
        count3 = 0
        for x2 in range(0, count2):
            count = 0
            for y in range(0, len(curr)):
                count += currW[count3]*curr[y]
                count3 += 1
            temp.append(t(A, count))
        curr = temp
    for l in range(0, len(curr)): curr[l] *= w_list[len(w_list) - 1][l]
    for l in curr: print(l, end=" ")
print(w)
print(i)
perceptron(a, i, w)
#Arav Singh, pd. 2, 2023