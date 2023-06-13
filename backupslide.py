import sys; args = sys.argv[1:]
arr = open(args[0]).read().splitlines() 
import time, math, random
# Arnav Kadam, pd. 6


global row, col, goal, invGoal, rowlColTable, nbrTable
row, col = 4,4

rowColTable = [(1,1), (1,2), (1,3), (1,4), (2,1), (2,2), (2,3), (2,4), (3,1), (3,2), (3,3), (3,4), (4,1), (4,2), (4,3), (4,4)]
nbrTable = [[1,4], [0, 2, 5], [1, 3, 6], [2,7], [0, 5, 8], [1, 4, 6, 9], [2, 5, 7, 10], [3, 6, 11], [4, 9, 12], [5, 8, 10, 13], [6, 9, 11, 14], [7, 10, 15], [8, 13], [9, 12, 14], [13, 10, 15], [11, 14]]


#the swap method swaps two characters in a puzzle string
def swap(puzzle, index1, index2):
    temp = []
    for item in puzzle:
        temp.append(item)
    temp[index1], temp[index2] = temp[index2], temp[index1]
    return "".join(temp)

def manhattan(puzzle):
   count = 0
   for char in puzzle:
      if char != "_":
         row1 = (puzzle.index(char) // row) + 1
         col1 = (puzzle.index(char) % col) + 1
         row2 = (goal.index(char) // row) + 1
         col2 = (goal.index(char) % col) + 1
         count+= abs(row1-row2) + abs(col1-col2)
   return count
   
def incrementalManhattan(parent, puzzle):
   char = parent[puzzle.index("_")]

   goalIndex = goal.index(char)
   goalRow = rowColTable[goalIndex][0]
   goalCol = rowColTable[goalIndex][1]

   parentIndex = parent.index(char)
   parentRow = rowColTable[parentIndex][0]
   parentCol = rowColTable[parentIndex][1]

   puzzleIndex = puzzle.index(char)
   puzzleRow = rowColTable[puzzleIndex][0]
   puzzleCol = rowColTable[puzzleIndex][1]

   if((abs(goalRow - parentRow) + abs(goalCol - parentCol)) < (abs(goalRow - puzzleRow) + abs(goalCol - puzzleCol))):
      return 1
   
   return -1



#the neighbors method returns a list of all of a puzzle's neighbors
#to do this, I find the row and column of the _ in the 2D visualization of the puzzle
#then, I use the swap method to generate the neighbors
#the row and column are not zero indexed

def neighbors(puzzle):
   index = puzzle.index("_")
   indexRow = rowColTable[index][0]
   indexCol = rowColTable[index][1]
   list = []
   if indexRow > 1:
      list.append(swap(puzzle, index, index-col))
   if indexRow < row:
      list.append(swap(puzzle, index, index+col))
   if indexCol > 1:
      list.append(swap(puzzle, index, index-1))
   if indexCol < col:
      list.append(swap(puzzle, index, index+1))
   return list

def neighbor(puzzle):
   index = puzzle.index("_")
   list = []
   for i in nbrTable[index]:
      list.append(swap(puzzle, index, i))
   return list



def inversionCount(puzzle):
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']
    temp = puzzle.replace("_", "")
    inversion = 0
    for i in range(len(temp)-1):
        for x in range(i, len(temp)):
            inversion+=alphabet.index(temp[i]) > alphabet.index(temp[x])
    return inversion

def impossiblePuzzle(puzzle):
    invCount = inversionCount(puzzle)
    if col % 2 != 0: return invCount % 2 != invGoal % 2
    else: return (invCount - abs((puzzle.index("")//row + 1)-(goal.index("")//row + 1))) % 2 != invGoal % 2

def findDir(puzzle, parent):
   index = puzzle.index("_")
   if parent.index("_") == index+col:
      return "U"
   if parent.index("_") == index-col:
      return "D"
   if parent.index("_") == index+1:
      return "L"
   if parent.index("_") == index-1:
      return "R"

def AStar(root):
   solved = [goal]
   path = []
   if(root == goal):
      return "G"
   if(impossiblePuzzle(root)):
      return "X"
   openSet = [[] for i in range(55)]
   tempMan = manhattan(root)
   openSet[tempMan].append((tempMan, root, 0, root, tempMan)) #(estimate, puzzle, level, parent, manhattan)
   min = tempMan
   index = 0
   closedSet = {root: [root, ""]}   #node --> parent, how parent got to the node
   while openSet:
      while not openSet[min]:
         min+=1
      dist, idealNbr, level, parent, oldManhat = openSet[min].pop()
      index+=1
      if not idealNbr in closedSet.keys():
         closedSet[idealNbr] = [parent, findDir(idealNbr, parent)]
      for nbr in neighbor(idealNbr):
         if nbr == goal:
            closedSet[goal] = [idealNbr, findDir(goal, idealNbr)]
            solved.append(closedSet.get(goal)[0])       # x --y      y -- c
            path.append(closedSet.get(goal)[1])
            temp = closedSet.get(goal)
            while solved[-1] != root:
               solved.append(closedSet.get(temp[0])[0])
               path.append(closedSet.get(temp[0])[1])
               temp = closedSet.get(temp[0])
            return "".join(path[::-1])
         if not nbr in closedSet and not nbr in openSet:
            change = incrementalManhattan(idealNbr, nbr) #1 or -1
            #print(change)
            estimate = (level+1) + (oldManhat + change)
            openSet[estimate].append((estimate, nbr, level+1, idealNbr, oldManhat+change))



#here is where I run all the necessary methods using the puzzle arguments passed in
#this also includes printing the number of steps and the time taken
start_time = time.process_time()
goal = arr[0]
invGoal = inversionCount(goal)
for i in range(0, len(arr)):
    path = AStar(arr[i])
    print(str(i) + ": " + arr[i] + " solved in " + str("{0:.3g}".format(time.process_time() - start_time)) + " secs with a path: " + path + ":")
    start_time = time.process_time()