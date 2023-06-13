import sys; args = sys.argv[1:]
import re
import math
#args = ["G7 V1,3,5R8", "G6W1 V4R8 V2B","G7 V3BR5", "G9 V4,5,7:B E6=7 V8R9 E5=8", "G9R9 V4,5,7B E4#1,3 E8#7,5 V8R"]
size,l,w = 0,0,0
pol = 0

graph = []
rewards = []
reward = False
def dimension(num):
    factors = []
    for i in range(1, num+1):
        if num % i == 0:
            factors.append(i)
        if num / i == math.sqrt(num):
            factors.append(i)
    index, minimum = 0, num+1
    for i in range(len(factors)-1):
        if factors[i+1] - factors[i] < minimum and factors[i+1] * factors[i] == num:
            index = i
            minimum = factors[i+1] - factors[i]
    l = min(factors[index+1], factors[index])
    w = max(factors[index+1], factors[index])
    return l,w

def equalChar(firstVal, secondVal):
    firstDir = 0
    secondDir = 0
    if firstVal == secondVal + w: firstDir, secondDir = 0, 2   # N 
    elif firstVal == secondVal - 1 : firstDir, secondDir = 1, 3  # E 
    elif firstVal == secondVal - w: firstDir, secondDir = 2, 0 # S 
    else: firstDir, secondDir = 3, 1 # W
    graph[firstVal][firstDir] = 1 if graph[firstVal][firstDir] == 0 else 0
    graph[secondVal][secondDir] = 1 if graph[secondVal][secondDir] == 0 else 0

def printGraph():
    for i in range(size):
        neighbors = []
        if i%w != 0:neighbors.append(i-1)
        if i > w-1: neighbors.append(i-w)
        if i%w != w-2: neighbors.append(i+1)
        if i < size - w: neighbors.append(i-w)

def findNeighbors(i):
    neighbors = []
    if i%w != 0: neighbors.append([(i-1), "E"])        #l = 1       w = 2
    if i > w-1: neighbors.append([(i-w), "S"])
    if (i+1)%w != 0: neighbors.append([(i+1), "W"])
    if i < size - w: neighbors.append([(i+w), "N"])
    return neighbors

def handleB(vals):
    realDirs = "NESW"
    directions = "SWNE"
    for x in vals:
        nbrs = findNeighbors(x)
        print(nbrs)
        print(w)
        for nbr, dir in nbrs:
            if nbr not in vals: 
                graph[x][directions.index(dir)] = 0
                graph[nbr][realDirs.index(dir)] = 0

def findNeighborsWithRes(i):
    neighbors = []
    if i%w != 0 and graph[i][3] != 0 and rewards[i-1] == 0: neighbors.append([(i-1), "E"])        #l = 1       w = 2
    if i > w-1 and graph[i][0] != 0 and rewards[i-w] == 0: neighbors.append([(i-w), "S"])
    if (i+1)%w != 0 and graph [i][1] != 0 and rewards[i+1] == 0: neighbors.append([(i+1), "W"])
    if i < size - w and graph[i][2] != 0 and rewards[i+w] == 0: neighbors.append([(i+w), "N"])
    return neighbors

def BFS():
    counter = 0
    overlord = []
    for i in range(size):
        if rewards[i] > 0:
            parseMe = [i] 
            seen = {i: [i, "", 0, 0]} #vertex, directions, distance, reward
            while parseMe:
                counter+=1
                curr = parseMe.pop(0)
                neighbors = findNeighborsWithRes(curr)
                for nbr, dir in neighbors:
                    if nbr!=i:
                        if nbr in seen:
                            distance = seen[nbr][2]
                            if seen[curr][2]+1 < distance: seen[nbr] = [nbr, dir, seen[curr][2]+1, 0]
                            if seen[curr][2]+1 == distance: seen[nbr][1] = seen[nbr][1] + dir
                        else:
                            parseMe.append(nbr)
                            seen[nbr] = [nbr, dir, seen[curr][2]+1, 0]
            #print(seen)
            overlord.append(seen)

    final = ["" for i in range(size)]
    for i in range(size):
        if rewards[i] > 0: final[i] = "*"
        else:
            minDist = 100
            dirs = ""
            for lst in overlord:
                if i in lst: 
                    if lst[i][2] < minDist: 
                        minDist = lst[i][2]
                        dirs = lst[i][1]
                    if lst[i][2] == minDist:
                        temp1 = {dirs}
                        temp2 = {*lst[i][1]}
                        temp3 = temp1.union(temp2)
                        if len(temp3) == len(dirs) + len(lst[i][1]): dirs+=lst[i][1]
            if not dirs: dirs = "."
            final[i] = dirs
    return final 
        
def printSolution(final):
    if reward == False:
        for i in range(size):
            print(".", end= "  ")
            if (i+1)%w == 0: 
                print()
        return
    for i in range(size):
        dirs = final[i]
        if rewards[i] > 0: print("*", end="  ")
        elif len(dirs) == 4: print("+", end = "  ")
        elif len(dirs) == 1: print(dirs, end = "  ")
        elif len(dirs) == 3:
            if "N" not in dirs: print("ˇ", end = "  ")
            if "S" not in dirs: print("^", end = "  ")
            if "W" not in dirs: print(">", end = "  ")
            if "E" not in dirs: print("<", end="  ")
        if "N" in dirs and "E" in dirs: print("L", end ="  ")
        elif "E" in dirs and "S" in dirs: print("r", end = "  ")
        elif "S" in dirs and "W" in dirs: print("7", end="  ")
        elif "W" in dirs and "N" in dirs: print("J", end ="  ")
        elif "N" in dirs and "S" in dirs: print("|", end="  ")
        elif "W" in dirs and "E" in dirs: print("-", end="  ")
        if (i+1)%w == 0: 
            print()
default = False
for arg in args:
    if arg[0] == "P":
        pol = int(arg[1])
    if arg[0] == "G":
        if "W" in arg and "R" in arg:
            size = int(arg[arg.index("G")+1:arg.index("W")])
            w = int(arg[arg.index("W")+1:arg[arg.index("R")]])
            l = size // w
            defaultR = int(arg[arg.index("R"):])
            default = True
        elif "W" in arg:
            size = int(arg[arg.index("G")+1:arg.index("W")])
            w = int(arg[arg.index("W")+1:])
            l = size // w
        elif "R" in arg:
            size = int(arg[arg.index("G")+1:arg.index("R")])
            defaultR = int(arg[arg.index("R")+1:])
            default = True
            l,w = dimension(size)
        else:
            size = int(arg[arg.index("G")+1:])
            l,w = dimension(size)
        for i in range(size): graph.append([1,1,1,1])  #NESW
        rewards = [0 for i in range(size)]
    if arg[0] == "V":
        if "R" in arg:
            if not reward: reward = True
            temp = arg[1:arg.index("R")].split(",")                
            if not default: reward = int(arg[arg.index("R")+1:])
            else: reward = defaultR
            for x in temp:
                if "B" in x: x = x.replace("B", "")
                if ":" in x:
                    for y in range(int(x[:len(x)-1]), size): reward[y] = reward
                else: rewards[int(x)] = reward
        if "B" in arg:
            temp = arg[1:arg.index("B")].split(",")
            vals = []
            for x in temp:
                if ":" in x:
                    for y in range(int(x[:len(x)-1]), size): vals.append(int(y))
                else: vals.append(int(x))
            print(vals)
            handleB(vals)
    if "E" in arg:
        if "=" in arg:
            firstVal = int(arg[arg.index("E")+1:arg.index("=")])
            secondVal = int(arg[arg.index("=")+1:]) 
            equalChar(firstVal, secondVal)
        if "#" in arg:
            firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("#")]).split(",")]
            secondVals = [int(val) for val in (arg[arg.index("#")+1:]).split(",")]
            for val1 in firstVals:
                for val2 in secondVals:
                    equalChar(val1, val2)

final = BFS()
finalString = ""
print(final)
for policy in final:
    if len(policy) == 1: finalString += policy 
    elif(policy == "NS"): finalString += "I"
    elif(policy == "NE"): finalString += "L"
    elif(policy == "NW"): finalString += "J"
    elif(policy == "SE"): finalString += "r"
    elif(policy == "SW"): finalString += "7"
    elif(policy == "EW"): finalString +="-"
    elif(policy == "NSW"): finalString += "<"
    elif(policy == "NSE"): finalString +=">"
    elif(policy == "NEW"): finalString += "^"
    elif(policy == "SEW"): finalString += "v"
    elif(policy == "NSEW"): finalString += "+"
print(finalString)