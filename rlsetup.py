import sys; args = sys.argv[1:]
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

def findNeighbors(x):
    neighbors = []
    if x % w != 0: neighbors.append([(x - 1), "R"])   
    if x > w-1: neighbors.append([(x - w), "D"])
    if (x + 1) % w != 0: neighbors.append([(x + 1), "L"])
    if x < size - w: neighbors.append([(x + w), "U"])
    return neighbors

def bChar(vals):
    realdirecs = "URDL"
    direcections = "DLUR"
    for x in vals:
        nbrs = findNeighbors(x)
        for nbr, direc in nbrs:
            if nbr not in vals: 
                graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
                graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1
def bChar2(vals, direc):
    realdirecs = "URDL"
    direcections = "DLUR"
    for x in vals:
        # nbrs = findNeighbors(x)
        # for nbr, direc in nbrs:
        #     if nbr not in vals: 
        if direc == "S": 
            direc = "U"
            nbr = x + w
        elif direc == "N": 
            direc = "D"
            nbr = x - w
        elif direc == "E": 
            direc = "L"
            nbr = x + 1
        elif direc == "W": 
            direc = "R"
            nbr = x - 1
        graph[x][direcections.index(direc)] = 0 if graph[x][direcections.index(direc)] != 0 else 1
        if nbr >= 0 and nbr < size: graph[nbr][realdirecs.index(direc)] = 0 if graph[nbr][realdirecs.index(direc)] != 0 else 1

def neighborstwo(x):
    neighbors = []
    if x % w != 0 and graph[x][3] != 0 and rewards[x - 1] == 0: neighbors.append([(x - 1), "R"])       
    if x > w - 1 and graph[x][0] != 0 and rewards[x - w] == 0: neighbors.append([(x - w), "D"])
    if (x + 1)% w != 0 and graph[x][1] != 0 and rewards[x + 1] == 0: neighbors.append([(x + 1), "L"])
    if x < size - w and graph[x][2] != 0 and rewards[x + w] == 0: neighbors.append([(x + w), "U"])
    return neighbors

def bfs():
    path_count = 0
    paths = []
    for i in range(size):
        if rewards[i] > 0:
            to_visit = [i]
            visited = {i: [i, "", 0, 0, rewards[i]]}
            while to_visit:
                curr = to_visit.pop(0)
                neighbors = neighborstwo(curr)
                for nbr, direc in neighbors:
                    if nbr != i:
                        if nbr in visited:
                            distance = visited[nbr][2]
                            currR = visited[nbr][4]
                            #if visited[curr][4] >= currR:
                            if visited[curr][2]+1 < distance:
                                visited[nbr] = [nbr, direc, visited[curr][2]+1, 0, visited[curr][4]]
                            if visited[curr][2]+1 == distance:
                                visited[nbr][1] += direc
                                #visited[nbr] = [nbr, visited[nbr][1]+direc, visited[curr][2]+1, 0]
                        else:
                            to_visit.append(nbr)
                            visited[nbr] = [nbr, direc, visited[curr][2]+1, 0, rewards[i]]
            paths.append(visited)
    if gCheck:
        for l in range(0, size):
            maxReward = 0
            for p in paths:
                if l in p and maxReward <= p[l][4]: maxReward = p[l][4]
            count = 0
            for p in paths:
                if l in p and p[l][4] != maxReward: paths[count].pop(l)
                count += 1
        final = ["." for i in range(size)]
    else:
        for l in range(0, size):
            maxReward = 0
            for p in paths:
                if l in p and p[l][2] != 0 and maxReward <= p[l][4] / p[l][2]: maxReward = p[l][4] / p[l][2]
            count = 0
            for p in paths:
                if l in p and p[l][2] != 0 and maxReward != p[l][4] / p[l][2]: paths[count].pop(l)
                count += 1
        final = ["." for i in range(size)]
    #mainLST = dict()
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
                        currReward = lst[i][4]
                        # mainL = ""
                        # maxReward = 0
                        # for x2 in mainLST:
                        #     temp = i
                        #     if lst[x2][2] == currDist:
                        #         print(lst[x2][1])
                        #         for x3 in lst[x2][1]:
                        #             if x3 == "U": 
                        #                 while rewards[temp] == 0: temp -= w
                        #             elif x3 == "D": 
                        #                 while rewards[temp] == 0: temp += w
                        #             elif x3 == "L": 
                        #                 while rewards[temp] == 0: temp -= 1
                        #             else: 
                        #                 while rewards[temp] == 0: temp += 1
                        #         if rewards[temp] > maxReward:
                        #             mainL = x2
                        #             maxReward = rewards[temp]
                        #print(mainL)
                        # totalDirec += lst[mainL][1]
                        for x2 in mainLST:
                            #print(mainLST)
                            #print(mainLST[x2])
                            if lst[x2][2] == currDist: totalDirec += lst[x2][1]
                        direcs += remove_duplicates(totalDirec)
            if len(mainLST) > 0 and direcs: 
                currReward = -1
                currRewards = []
                for l in mainLST:
                    for m in mainLST[l]:
                        if m[2] == min_dist and m[4] >= currReward: 
                            if m[4] > currReward:
                                currReward = m[4]
                                currRewards = [m]
                            if m[4] == currReward: 
                                currRewards.append(m)
            #print(currRewards)
            if len(mainLST) > 0 and direcs: 
                direcs = ""
                for c in currRewards: direcs += c[1]
                # for l in mainLST:
                #     for m in mainLST[l]:
                #         if m[2] == min_dist and m in currRewards: 
                #             if direcs: 
                #                 direcs = m[1]
                                #break
                #print(mainLST, min_dist)
            if not direcs:
                direcs = "."
            final[i] = direcs
            path_count += len(direcs)

    return final
default = False
count = 0
gCheck = False
for a in args:
    if "G" in a: 
        gCheck = True
        break
if len(args) == 1: print("."*int(args[0]))
elif len(args) == 2 and "G" in args[1]: 
    policy = ""
    for x in range(0, int(args[0])): policy += "."
    print(policy)
elif len(args) == 3 and args[1].isnumeric():
    policy = ""
    for x in range(0, int(args[0])): policy += "."
    print(policy)
else:
    for arg in args:
        if count == 0:
            if args[1].isnumeric(): 
                size = int(arg)
                w = int(args[1])
                l = size // w
                for i in range(size): graph.append([1,1,1,1]) 
                rewards = [0 for i in range(size)]
            else:
                size = int(arg)
                default = True
                l,w = dim(size)
                for i in range(size): graph.append([1,1,1,1]) 
                rewards = [0 for i in range(size)]
        if arg[0] == "R" or arg[0] == "r":
            if ":" not in arg: rewards[int(arg[1:])] = 12
            else: 
                if arg[1] == ":": rewards[0] = int(arg[arg.index(":") + 1:])
                else: rewards[int(arg[1:arg.index(":")])] = int(arg[arg.index(":") + 1:])
        #if arg[0] == "G": print("hi")
        if arg[0] == "B" or arg[0] == "b":
            if "N" in arg or "S" in arg or "E" in arg or "W" in arg:
                postLet = arg[-1]
                fVal = int(arg[arg.index("B")+1:len(arg)-1])
                bChar2([fVal], postLet)
            else:
                fVal = int(arg[arg.index("B")+1:])
                bChar([fVal])
        count += 1
    #print(rewards)
    final = bfs()
    #print(final) 
    # print(graph)
    finalString = ""
    for policy in final:
        policy = remove_duplicates(policy)
        if len(policy) == 1: finalString += policy 
        elif(policy == "UD"): finalString += "|"
        elif(policy == "UR"): finalString += "V"
        elif(policy == "UL"): finalString += "M"
        elif(policy == "DR"): finalString += "S"
        elif(policy == "DL"): finalString += "E"
        elif(policy == "DU"): finalString += "|"
        elif(policy == "RU"): finalString += "V"
        elif(policy == "LU"): finalString += "M"
        elif(policy == "RD"): finalString += "S"
        elif(policy == "LD"): finalString += "E"
        elif(policy == "RL"): finalString +="-"
        elif(policy == "LR"): finalString +="-"
        elif(len(policy) == 4): finalString += "+"
        elif(len(policy) == 3):
            if "U" not in policy: finalString +="T"
            if "D" not in policy: finalString +="N"
            if "L" not in policy: finalString +="W"
            if "R" not in policy: finalString +="F"
    print(finalString)
#Arav Singh, pd. 2, 2023