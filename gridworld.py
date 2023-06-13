import sys; args = sys.argv[1:]
import re
import math
size,l,w = 0,0,0
pol = 0

graph = []
rewards = []
reward = False
def remove_duplicates(string):
    unique_chars = []
    for char in string:
        if char not in unique_chars:
            unique_chars.append(char)
    return ''.join(unique_chars)
def dim(num):
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
    # factors = []
    # for i in range(1, int(math.sqrt(num))+1):
    #     if num % i == 0:
    #         factors.append(i)
    #         if i != num//i:
    #             factors.append(num//i)
    # min_diff = num
    # for i in range(len(factors)):
    #     for j in range(i+1, len(factors)):
    #         if factors[i] * factors[j] == num:
    #             diff = abs(factors[i] - factors[j])
    #             if diff < min_diff:
    #                 min_diff = diff
    #                 l, w = factors[i], factors[j]
    # if l > w:
    #     l, w = w, l     
    # return l, w

def tildaChar(firstVal, secondVal):
    firstdirec = 0
    if firstVal == secondVal + w: firstdirec = 0   
    elif firstVal == secondVal - 1: firstdirec = 1  
    elif firstVal == secondVal - w: firstdirec = 2 
    else: firstdirec = 3 
    graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0

def equalChar(firstVal, secondVal):
    firstdirec = 0
    seconddirec = 0
    if firstVal == secondVal + w: firstdirec, seconddirec = 0, 2    
    elif firstVal == secondVal - w: firstdirec, seconddirec = 2, 0 
    elif firstVal == secondVal - 1 : firstdirec, seconddirec = 1, 3 
    else: firstdirec, seconddirec = 3, 1 
    graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0
    graph[secondVal][seconddirec] = 1 if graph[secondVal][seconddirec] == 0 else 0

def tildaChar2(firstVal, secondVal, postlet):
    firstdirec = 0
    seconddirec = 0
    # if firstVal == secondVal + w: firstdirec, seconddirec = 0, 2    
    # elif firstVal == secondVal - w: firstdirec, seconddirec = 2, 0 
    # elif firstVal == secondVal - 1 : firstdirec, seconddirec = 1, 3 
    if postlet == "S": firstdirec, seconddirec = 0, 2    
    elif postlet == "N": firstdirec, seconddirec = 2, 0 
    elif postlet == "W": firstdirec, seconddirec = 1, 3 
    else: firstdirec, seconddirec = 3, 1 
    graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0
    # firstdirec = 0
    # # if firstVal == secondVal + w: firstdirec = 0   
    # # elif firstVal == secondVal - 1: firstdirec = 1  
    # # elif firstVal == secondVal - w: firstdirec = 2 
    # if postlet == "S": firstdirec = 0   
    # elif postlet == "W": firstdirec = 1  
    # elif postlet == "N": firstdirec = 2 
    # else: firstdirec = 3 
    # graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0

def equalChar2(firstVal, secondVal, postlet):
    firstdirec = 0
    seconddirec = 0
    # if firstVal == secondVal + w: firstdirec, seconddirec = 0, 2    
    # elif firstVal == secondVal - w: firstdirec, seconddirec = 2, 0 
    # elif firstVal == secondVal - 1 : firstdirec, seconddirec = 1, 3 
    if postlet == "S": firstdirec, seconddirec = 0, 2    
    elif postlet == "N": firstdirec, seconddirec = 2, 0 
    elif postlet == "W": firstdirec, seconddirec = 1, 3 
    else: firstdirec, seconddirec = 3, 1 
    graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0
    graph[secondVal][seconddirec] = 1 if graph[secondVal][seconddirec] == 0 else 0
    # graph[firstVal][firstdirec] = 1 if graph[firstVal][firstdirec] == 0 else 0
    # graph[secondVal][seconddirec] = 1 if graph[secondVal][seconddirec] == 0 else 0

def findNeighbors(x):
    neighbors = []
    if x % w != 0: neighbors.append([(x - 1), "E"])   
    if x > w-1: neighbors.append([(x - w), "S"])
    if (x + 1) % w != 0: neighbors.append([(x + 1), "W"])
    if x < size - w: neighbors.append([(x + w), "N"])
    return neighbors

# def bChar(vals):
#     realdirecs = "NESW"
#     direcections = "SWNE"
#     for x in vals:
#         nbrs = findNeighbors(x)
#         for nbr, direc in nbrs:
#             if nbr not in vals: 
#                 graph[x][direcections.index(direc)] = 0
#                 graph[nbr][realdirecs.index(direc)] = 0

def bChar(vals):
    realdirecs = "NESW"
    direcections = "SWNE"
    for x in vals:
        nbrs = findNeighbors(x)
        for nbr, direc in nbrs:
            if nbr not in vals: 
                graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
                graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1

def bChar2(x, nbr, direc):
    realdirecs = "URDL"
    direcections = "DLUR"
    #for x in vals:
        # nbrs = findNeighbors(x)
        # for nbr, direc in nbrs:
        #     if nbr not in vals: 
    if direc == "S": 
        direc = "U"
        #nbr = x + w
    elif direc == "N": 
        direc = "D"
        # nbr = x - w
    elif direc == "E": 
        direc = "L"
        # nbr = x + 1
    elif direc == "W": 
        direc = "R"
        # nbr = x - 1
    graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
    if nbr >= 0 and nbr < size: graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1

def bChar3(x, nbr, direc):
    realdirecs = "URDL"
    direcections = "DLUR"
    if direc == "S": 
        direc = "U"
        #nbr = x + w
    elif direc == "N": 
        direc = "D"
        #nbr = x - w
    elif direc == "E": 
        direc = "L"
        #nbr = x + 1
    elif direc == "W": 
        direc = "R"
       # nbr = x - 1
    graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
        #if nbr >= 0 and nbr < size: graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1

def bChar4(x, nbr):
    realdirecs = "URDL"
    direcections = "DLUR"
    #for x in vals:
        # nbrs = findNeighbors(x)
        # for nbr, direc in nbrs:
        #     if nbr not in vals: 
    if x == nbr - w: 
        direc = "U"
        #nbr = x + w
    elif x == nbr + w: 
        direc = "D"
        # nbr = x - w
    elif x == nbr - 1: 
        direc = "L"
        # nbr = x + 1
    elif x == nbr + 1: 
        direc = "R"
        # nbr = x - 1
    print(x, nbr, w)
    graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
    if nbr >= 0 and nbr < size: graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1

def bChar5(x, nbr):
    realdirecs = "URDL"
    direcections = "DLUR"
    if x == nbr - w: 
        direc = "U"
        #nbr = x + w
    elif x == nbr + w: 
        direc = "D"
        # nbr = x - w
    elif x == nbr - 1: 
        direc = "L"
        # nbr = x + 1
    elif x == nbr + 1: 
        direc = "R"
        # nbr = x - 1
    graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1

def neighborstwo(x):
    neighbors = []
    if x % w != 0 and graph[x][3] != 0 and rewards[x - 1] == 0: neighbors.append([(x - 1), "E"])       
    if x > w - 1 and graph[x][0] != 0 and rewards[x - w] == 0: neighbors.append([(x - w), "S"])
    if (x + 1)% w != 0 and graph[x][1] != 0 and rewards[x + 1] == 0: neighbors.append([(x + 1), "W"])
    if x < size - w and graph[x][2] != 0 and rewards[x + w] == 0: neighbors.append([(x + w), "N"])
    return neighbors

def bfs():
    path_count = 0
    paths = []
    for i in range(size):
        if rewards[i] > 0:
            to_visit = [i]
            visited = {i: [i, "", 0, 0]}
            while to_visit:
                curr = to_visit.pop(0)
                neighbors = neighborstwo(curr)
                for nbr, direc in neighbors:
                    if nbr != i:
                        if nbr in visited:
                            distance = visited[nbr][2]
                            if visited[curr][2]+1 < distance:
                                visited[nbr] = [nbr, direc, visited[curr][2]+1, 0]
                            if visited[curr][2]+1 == distance:
                                visited[nbr][1] += direc
                                #visited[nbr] = [nbr, visited[nbr][1]+direc, visited[curr][2]+1, 0]
                        else:
                            to_visit.append(nbr)
                            visited[nbr] = [nbr, direc, visited[curr][2]+1, 0]
            paths.append(visited)

    final = ["." for i in range(size)]
    for i in range(size):
        if rewards[i] > 0:
            final[i] = "*"
        else:
            min_dist = float('inf')
            direcs = ""
            mainLST = dict()
            for lst in paths:
                if i in lst:
                    if lst[i][2] < min_dist:
                        min_dist = lst[i][2]
                        direcs = lst[i][1]
                    if lst[i][2] == min_dist:
                        if i not in mainLST: mainLST[i] = [lst[i]]
                        else: 
                            f = mainLST[i]
                            f.append(lst[i])
                            mainLST[i] = f
                        totalDirec = lst[i][1]
                        currDist = lst[i][2]
                        for x2 in mainLST:
                            #print(mainLST)
                            if lst[x2][2] == currDist: totalDirec += lst[x2][1]
                        direcs += remove_duplicates(totalDirec)
                        # temp1 = set(direcs)
                        # temp2 = set(lst[i][1])
                        # if len(temp1.union(temp2)) == len(direcs) + len(lst[i][1]): direcs += lst[i][1]
            if not direcs:
                direcs = "."
            final[i] = direcs
            path_count += len(direcs)

    return final
      
default = False
edgePairs = []
for arg in args:
    if arg[0:2] == "GG": arg = arg[1:]
    if "BB" in arg: arg = arg.replace("BB", "B")
    if arg[0] == "P": pol = int(arg[1])
    if arg[0] == "G":
        if "W" in arg and "R" in arg:
            size = int(arg[arg.index("G")+1:arg.index("W")])
            w = int(arg[arg.index("W")+1:arg.index("R")])
            l = size // w
            defaultR = int(arg[arg.index("R")+1:])
            default = True
        elif "W" in arg:
            size = int(arg[arg.index("G")+1:arg.index("W")])
            w = int(arg[arg.index("W")+1:])
            l = size // w
        elif "R" in arg:
            size = int(arg[arg.index("G")+1:arg.index("R")])
            defaultR = int(arg[arg.index("R")+1:])
            default = True
            l,w = dim(size)
        else:
            size = int(arg[arg.index("G")+1:])
            l,w = dim(size)
        for i in range(size): graph.append([1,1,1,1]) 
        rewards = [0 for i in range(size)]
    if arg[0] == "V":
        if "R" in arg:
            bCheck = True
            vals = []
            if arg[-1] == "B": 
                bCheck = False
                #arg = arg[0:len(arg) - 1]
                temp = arg[1:arg.index("R")].split(",")
                for x in temp:
                    if ":" in x:
                        for y in range(int(x[:len(x)-1]), size): vals.append(int(y))
                    else: vals.append(int(x))
                if not bCheck: arg = arg[0:len(arg) - 1]
            if not reward: reward = True
            temp = arg[1:arg.index("R")].split(",")                
            if not default: reward = int(arg[arg.index("R")+1:])
            else: reward = defaultR
            for x in temp:
                if "B" in x: x = x.replace("B", "")
                if ":" in x:
                    for y in range(int(x[:len(x)-1]), size): 
                        rewards[y] = reward
                        # for e in edgePairs: 
                        #     if e[0] == y: rewards[e[1]] = reward
                        #     if e[1] == y: rewards[e[0]] = reward
                else: 
                    rewards[int(x)] = reward
                    # for e in edgePairs: 
                    #     if e[0] == int(x): rewards[e[1]] = reward
                    #     if e[1] == int(x): rewards[e[0]] = reward
            bChar(vals)
        if "B" in arg:
            temp = arg[1:arg.index("B")].split(",")
            vals = []
            for x in temp:
                if ":" in x:
                    for y in range(int(x[:len(x)-1]), size): vals.append(int(y))
                else: vals.append(int(x))
            bChar(vals)
    if arg[0] == "E":
        #if "::" in arg: arg = arg.replace("::", ",")
        if "," in arg or ":" in arg or "::" in arg and "R" not in arg and "T" not in arg and "=" != arg[-1] and "~" != arg[-1] and "#" not in arg:
            if "=" in arg:
                if "," in arg:
                    firstVals = (arg[arg.index("E")+1:arg.index("=")]).split(",")
                    secondVals = (arg[arg.index("=")+1:]).split(",")
                    for x2 in range(0, len(firstVals)):
                        if [int(firstVals[x2]), int(secondVals[x2]), True] not in edgePairs or [int(secondVals[x2]), int(firstVals[x2]), True] not in edgePairs: edgePairs.append([int(firstVals[x2]), int(secondVals[x2]), True])
                        else: 
                            for x in range(0, len(edgePairs)):
                                if edgePairs[x] == [int(firstVals[x2]), int(secondVals[x2]), True] or edgePairs[x] == [int(secondVals[x2]), int(firstVals[x2]), True]: 
                                    edgePairs[x] = [int(firstVals[x2]), int(secondVals[x2]), False]
                                    break
                        bChar4(int(firstVals[x2]), int(secondVals[x2]))
                elif "::" in arg:
                    midSfirst = arg[arg.index("E")+1:arg.index("=")].split("::")
                    mid = (arg[arg.index("=")+1:]).split("::")
                    m = int(midSfirst[0])
                    firstVals = []
                    stepper = int(midSfirst[1])
                    m2 = int(mid[0])
                    secondVals = []
                    stepper2 = int(mid[1])
                    while m < size: 
                        firstVals.append(m)
                        m += stepper
                    while m2 < size: 
                        secondVals.append(m2)
                        m2 += stepper2
                    for x2 in range(0, len(firstVals)):
                        if [int(firstVals[x2]), int(secondVals[x2]), True] not in edgePairs or [int(secondVals[x2]), int(firstVals[x2]), True] not in edgePairs: edgePairs.append([int(firstVals[x2]), int(secondVals[x2]), True])
                        else: 
                            for x in range(0, len(edgePairs)):
                                if edgePairs[x] == [int(firstVals[x2]), int(secondVals[x2]), True] or edgePairs[x] == [int(secondVals[x2]), int(firstVals[x2]), True]: 
                                    edgePairs[x] = [int(firstVals[x2]), int(secondVals[x2]), False]
                                    break
                        bChar4(int(firstVals[x2]), int(secondVals[x2]))
            if "~" in arg:
                firstVals = (arg[arg.index("E")+1:arg.index("~")]).split(",")
                secondVals = (arg[arg.index("~")+1:]).split(",")
                for x2 in range(0, len(firstVals)):
                    bChar5(int(firstVals[x2]), int(secondVals[x2]))
        elif "=" == arg[-1] or "~" == arg[-1]:
            postLet = arg[-2]
            if ":" not in arg and "::" not in arg:
                fVal = int(arg[arg.index("E")+1:len(arg)-2])
                if postLet == "N": sVal = fVal - w
                    # temp = fVal
                    # while temp > 0: 
                    #     sVal.append(temp - w)
                    #     temp -= w
                elif postLet == "S": sVal = fVal + w
                    # temp = fVal
                    # while temp < size - w: 
                    #     sVal.append(temp + w)
                    #     temp += w
                elif postLet == "W": sVal = fVal - 1
                    # temp = fVal
                    # while temp % w > 0: 
                    #     sVal.append(temp - 1)
                    #     temp -= 1
                else: sVal = fVal + 1
                    # temp = fVal
                    # while (temp + 1) % w == 0: 
                    #     sVal.append(temp+1)
                    #     temp += 1
                if arg[-1] == "~": bChar3(fVal, sVal, postLet)
                    #for s in sVal: bChar3(fVal, s, postLet)
                else: 
                    if [fVal, sVal, True] not in edgePairs or [sVal, fVal, True] not in edgePairs: edgePairs.append([fVal, sVal, True])
                    else: 
                        for x in range(0, len(edgePairs)):
                            if edgePairs[x] == [sVal, fVal, True] or edgePairs[x] == [fVal, sVal, True]: 
                                edgePairs[x] = [fVal, sVal, False]
                                break
                    bChar2(fVal, sVal, postLet)
                    #for s in sVal: equalChar2(fVal, s, postLet)
            elif ":" in arg and "::" not in arg:
                midSfirst = arg[arg.index("E")+1:len(arg)-2].split(":")
                midS = [x for x in range(int(midSfirst[0]), int(midSfirst[1]))]
                for a in midS:
                    fVal = a
                    if postLet == "N": sVal = fVal - w
                        # temp = fVal
                        # while temp > 0: 
                        #     sVal.append(temp - w)
                        #     temp -= w
                    elif postLet == "S": sVal = fVal + w
                        # temp = fVal
                        # while temp < size - w: 
                        #     sVal.append(temp + w)
                        #     temp += w
                    elif postLet == "W": sVal = fVal - 1
                        # temp = fVal
                        # while temp % w > 0: 
                        #     sVal.append(temp - 1)
                        #     temp -= 1
                    else: sVal = fVal + 1
                        # temp = fVal
                        # while (temp + 1) % w == 0: 
                        #     sVal.append(temp+1)
                        #     temp += 1
                    if arg[-1] == "~": bChar3(fVal, sVal, postLet)
                        #for s in sVal: bChar3(fVal, s, postLet)
                    else: 
                        if [fVal, sVal, True] not in edgePairs or [sVal, fVal, True] not in edgePairs: edgePairs.append([fVal, sVal, True])
                        else: 
                            for x in range(0, len(edgePairs)):
                                if edgePairs[x] == [sVal, fVal, True] or edgePairs[x] == [fVal, sVal, True]: 
                                    edgePairs[x] = [fVal, sVal, False]
                                    break
                        bChar2(fVal, sVal, postLet)
                        #for s in sVal: equalChar2(fVal, s, postLet)
            elif "::" in arg:
                midSfirst = arg[arg.index("E")+1:len(arg)-2].split("::")
                m = int(midSfirst[0])
                midS = []
                stepper = int(midSfirst[1])
                while m < size: 
                    midS.append(m)
                    m += stepper
                #midS = [x for x in range(int(midSfirst[0]), int(midSfirst[1]))]
                for a in midS:
                    fVal = a
                    if postLet == "N": sVal = fVal - w
                        # temp = fVal
                        # while temp > 0: 
                        #     sVal.append(temp - w)
                        #     temp -= w
                    elif postLet == "S": sVal = fVal + w
                        # temp = fVal
                        # while temp < size - w: 
                        #     sVal.append(temp + w)
                        #     temp += w
                    elif postLet == "W": sVal = fVal - 1
                        # temp = fVal
                        # while temp % w > 0: 
                        #     sVal.append(temp - 1)
                        #     temp -= 1
                    else: sVal = fVal + 1
                        # temp = fVal
                        # while (temp + 1) % w == 0: 
                        #     sVal.append(temp+1)
                        #     temp += 1
                    if arg[-1] == "~": bChar3(fVal, sVal, postLet)
                        #for s in sVal: bChar3(fVal, s, postLet)
                    else: 
                        if [fVal, sVal, True] not in edgePairs or [sVal, fVal, True] not in edgePairs: edgePairs.append([fVal, sVal, True])
                        else: 
                            for x in range(0, len(edgePairs)):
                                if edgePairs[x] == [sVal, fVal, True] or edgePairs[x] == [fVal, sVal, True]: 
                                    edgePairs[x] = [fVal, sVal, False]
                                    break
                        bChar2(fVal, sVal, postLet)
        elif "=" in arg and "," in arg:
            if "R" in arg:
                firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("=")]).split(",")]
                secondVals = [int(val) for val in (arg[arg.index("=")+1:arg.index("R")]).split(",")]
                for val1 in firstVals:
                    for val2 in secondVals:
                        bChar4(val1, val2)
                if not reward: reward = True               
                if not default: reward = int(arg[arg.index("R")+1:])
                else: reward = defaultR
                for x2 in firstVals: rewards[x2] = reward
                for x2 in secondVals: rewards[x2] = reward
            else:
                firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("=")]).split(",")]
                secondVals = [int(val) for val in (arg[arg.index("=")+1:]).split(",")]
                for val1 in firstVals:
                    for val2 in secondVals:
                        if [val1, val2, True] not in edgePairs or [val2, val1, True] not in edgePairs: edgePairs.append([val1, val2, True])
                        else: 
                            for x in range(0, len(edgePairs)):
                                if edgePairs[x] == [val2, val1, True] or edgePairs[x] == [val1, val2, True]: 
                                    edgePairs[x] = [val1, val2, False]
                                    break
                        bChar4(val1, val2)
        elif "~" in arg and "," in arg:
            if "R" in arg:
                firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("~")]).split(",")]
                secondVals = [int(val) for val in (arg[arg.index("~")+1:arg.index("R")]).split(",")]
                for val1 in firstVals:
                    for val2 in secondVals:
                        bChar5(val1, val2)
                if not reward: reward = True               
                if not default: reward = int(arg[arg.index("R")+1:])
                else: reward = defaultR
                for x2 in firstVals: rewards[x2] = reward
                for x2 in secondVals: rewards[x2] = reward
            else:
                firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("~")]).split(",")]
                secondVals = [int(val) for val in (arg[arg.index("~")+1:]).split(",")]
                for val1 in firstVals:
                    for val2 in secondVals:
                        bChar5(val1, val2)
        elif "=" in arg:
            if "R" in arg:
                firstVal = int(arg[arg.index("E")+1:arg.index("=")])
                secondVal = int(arg[arg.index("=")+1:arg.index("R")]) 
                if [firstVal, secondVal, True] not in edgePairs or [secondVal, firstVal, True] not in edgePairs: edgePairs.append([firstVal, secondVal, True])
                else: 
                    for x in range(0, len(edgePairs)):
                        if edgePairs[x] == [secondVal, firstVal, True] or edgePairs[x] == [firstVal, secondVal, True]: 
                            edgePairs[x] = [firstVal, secondVal, False]
                            break
                bChar4(firstVal, secondVal)
                if not reward: reward = True               
                if not default: reward = int(arg[arg.index("R")+1:])
                else: reward = defaultR
                rewards[firstVal] = reward
                rewards[secondVal] = reward
            else: 
                firstVal = int(arg[arg.index("E")+1:arg.index("=")])
                secondVal = int(arg[arg.index("=")+1:]) 
                if [firstVal, secondVal, True] not in edgePairs or [secondVal, firstVal, True] not in edgePairs: edgePairs.append([firstVal, secondVal, True])
                else: 
                    for x in range(0, len(edgePairs)):
                        if edgePairs[x] == [secondVal, firstVal, True] or edgePairs[x] == [firstVal, secondVal, True]: 
                            edgePairs[x] = [firstVal, secondVal, False]
                            break
                bChar4(firstVal, secondVal)
        elif "~" in arg:
            if "R" in arg:
                firstVal = int(arg[arg.index("E")+1:arg.index("~")])
                secondVal = int(arg[arg.index("~")+1:arg.index("R")]) 
                bChar5(firstVal, secondVal)
                if not reward: reward = True               
                if not default: reward = int(arg[arg.index("R")+1:])
                else: reward = defaultR
                rewards[firstVal] = reward
                rewards[secondVal] = reward
            else:
                firstVal = int(arg[arg.index("E")+1:arg.index("~")])
                secondVal = int(arg[arg.index("~")+1:]) 
                bChar5(firstVal, secondVal)
        elif "#" in arg:
            # firstVal = int(arg[arg.index("E")+1:arg.index("#")])
            # secondVal = int(arg[arg.index("#")+1:]) 
            # equalChar(firstVal, secondVal)
            firstVals = [int(val) for val in (arg[arg.index("E")+1:arg.index("#")]).split(",")]
            secondVals = [int(val) for val in (arg[arg.index("#")+1:]).split(",")]
            for val1 in firstVals:
                for val2 in secondVals:
                    if [val1, val2, True] not in edgePairs or [val2, val1, True] not in edgePairs: edgePairs.append([val1, val2, True])
                    else: 
                        for x in range(0, len(edgePairs)):
                            if edgePairs[x] == [val2, val1, True] or edgePairs[x] == [val1, val2, True]: 
                                edgePairs[x] = [val1, val2, False]
                                break
                    bChar4(val1, val2)

final = bfs()
print(final)
print(edgePairs)
finalString = ""
for policy in final:
    policy = remove_duplicates(policy)
    if len(policy) == 1: finalString += policy 
    elif(policy == "NS"): finalString += "|"
    elif(policy == "NE"): finalString += "L"
    elif(policy == "NW"): finalString += "J"
    elif(policy == "SE"): finalString += "r"
    elif(policy == "SW"): finalString += "7"
    elif(policy == "SN"): finalString += "|"
    elif(policy == "EN"): finalString += "L"
    elif(policy == "WN"): finalString += "J"
    elif(policy == "ES"): finalString += "r"
    elif(policy == "WS"): finalString += "7"
    elif(policy == "EW"): finalString +="-"
    elif(policy == "WE"): finalString +="-"
    elif(len(policy) == 4): finalString += "+"
    elif(len(policy) == 3):
        if "N" not in policy: finalString +="v"
        if "S" not in policy: finalString +="^"
        if "W" not in policy: finalString +=">"
        if "E" not in policy: finalString +="<"
print("Policy:",finalString)
#Arav Singh, Pd. 2, 2023