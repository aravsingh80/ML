import numpy as np
from board import Board
#from intelligence import a_star, manhattan_heuristic, n_wrong_heuristic
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from heapq import heappush, heappop, heapify
from collections import deque
import math
import random
def GoalTest(v):
    s = ""
    for x in v:
        s += x
    return s == find_goal(s)

def print_puzzle(x, y):
    s = ""
    for x2 in range(0, len(y)):
        s += y[x2]
        if x2 % int(x) == (int(x) - 1):
            print(s)
            s = ""

def nonOrderPairCount(board, size):
    count = 0
    for x in range(0, size):
        for y in range(x + 1, size):
            if board[x] != "." and board[y] != ".":
                if board[x] > board[y]:
                    count += 1
    return count
                

def parity(board, size):
    m = []
    y = []
    non = nonOrderPairCount(board, len(board))
    for x2 in range(0, len(board)):
        y.append(board[x2])
        if x2 % int(size) == (int(size) - 1):
            m.append(y)
            y = []
        x = 0
    y = 0
    b = False
    size = int(size)
    for x2 in range(0, size):
        for x3 in range(0, size):
            if m[x2][x3] == '.':
                x = x2
                y = x3
                b = True
            if b:
                break
        if b:
            break
    if size % 2 == 0:
        if x % 2 == 0 and non % 2 == 1:
            return True
        elif x % 2 == 1 and non % 2 == 0:
            return True
        else:
            return False
    else:
        if non % 2 == 0:
            return True
        else:
            return False
def find_goal(board):
    s = ""
    for x in board:
        if x != ".":
            s += x
    s = ''.join(sorted(s)) 
    s += "."
    return s

def swap(x, y, x2, y2, m):
    m2 = m
    temp = m2[x][y]
    m2[x][y] = m2[x2][y2]
    m2[x2][y2] = temp
    return m2

def tostr(m):
    temp = ""
    for x in range(len(m)):
            temp += ''.join(m[x])
    return temp

def directionchildren(board, size):
    m5 = []
    y = []
    m9 = []
    for x2 in range(0, len(board)):
        y.append(board[x2])
        if x2 % int(size) == (int(size) - 1):
            m5.append(y)
            y = []
    boardset = set()
    x = 0
    y = 0
    b = False
    size = int(size)
    for x2 in range(0, size):
        for x3 in range(0, size):
            if m5[x2][x3] == '.':
                x = x2
                y = x3
                b = True
            if b:
                break
        if b:
            break
    if x > 0:
        temp = swap(x, y, x - 1, y, m5)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m9.append("U")
        m5 = swap(x, y, x - 1, y, m5)
    if y > 0:
        temp = swap(x, y, x, y - 1, m5)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m9.append("L")
        m5 = swap(x, y, x, y - 1, m5)
    if x < size - 1:
        temp = swap(x, y, x + 1, y, m5)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m9.append("D")
        m5 = swap(x, y, x + 1, y, m5)
    if y < size - 1:
        temp = swap(x, y, x, y + 1, m5)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m9.append("R")
        m5 = swap(x, y, x, y + 1, m5)
    return m9

def get_children(board, size):
    m = []
    y = []
    for x2 in range(0, len(board)):
        y.append(board[x2])
        if x2 % int(size) == (int(size) - 1):
            m.append(y)
            y = []
    boardset = set()
    x = 0
    y = 0
    b = False
    size = int(size)
    for x2 in range(0, size):
        for x3 in range(0, size):
            if m[x2][x3] == '.':
                x = x2
                y = x3
                b = True
            if b:
                break
        if b:
            break
    if x > 0:
        temp = swap(x, y, x - 1, y, m)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m = swap(x, y, x - 1, y, m)
    if y > 0:
        temp = swap(x, y, x, y - 1, m)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m = swap(x, y, x, y - 1, m)
    if x < size - 1:
        temp = swap(x, y, x + 1, y, m)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m = swap(x, y, x + 1, y, m)
    if y < size - 1:
        temp = swap(x, y, x, y + 1, m)
        temp2 = tostr(temp)
        boardset.add(temp2)
        m = swap(x, y, x, y + 1, m)
    return boardset

def findCoordinate(value, board, size):
    x = board.index(value)
    return (x / size, x % size)

def bfs(startnode, size):
    fringe = deque()
    visited = set()
    fringe.append((startnode, 0))
    visited.add(startnode)
    while len(fringe) > 0:
        v, moves = fringe.popleft()
        if GoalTest(v):
            return moves
        for c in get_children(v, size):
            if c not in visited:
                fringe.append((c, moves + 1))
                visited.add(c)
    return None

def scrambleBoard(board):
    board = list(board)
    random.shuffle(board)
    return ''.join(board)

def heuristic(startstate):
    size = int(len(startstate) ** 0.5)
    count = 0
    for x in startstate:
        if x != ".":
            y, y1 = findCoordinate(x, startstate, size)
            z, z1 = findCoordinate(x, find_goal(startstate), size)
            y2 = abs(y - z)
            z2 = abs(y1 - z1)
            count += y2 + z2
    return count 
import time
def a_star(startstate):
    closed = set()
    startnode = (heuristic(startstate), 0, startstate)      
    fringe = []
    heappush(fringe, startnode)
    while len(fringe) > 0:
        f, depth, v = heappop(fringe)
        if GoalTest(v):
            return depth 
        if v not in closed:
            closed.add(v)
            for c in get_children(v, int(len(startstate) ** 0.5)):
                if c not in closed:
                    temp = (depth + 1 + heuristic(c), depth + 1, c)
                    heappush(fringe, temp)
    return None
np.random.seed(1)
"""
setup of the network
input (256) -> fully connected (512) -> fully connected (1024) -> fully connected (512) -> output (1)
using dropout along the way to avoid overfitting
"""

model = Sequential()
model.add(Dense(units=512, input_dim=256, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=1024, activation='relu'))
model.add(Dropout(0.1))
model.add(Dense(units=512, activation='relu'))

model.add(Dense(units=1, activation='linear'))

model.compile(optimizer='adam',
              loss='mse')


# model = load_model("models/neural_network - 512x1024x512 - 35.h5") # use this to load the model


def transform(state):
    """
    just a helper function to transform the game state into a 256 element numpy array
    this way is best for the neural net since it normally doesn't understand that 15 and 14 are totally disparate
    """
    vector_state = state.flatten()

    output = []
    for i in range(16):
        one_hot = np.zeros(16)
        one_hot[np.argwhere(vector_state == i)] = 1

        output.append(one_hot)

    return np.array(output).flatten().reshape(256)


def training_data(n_boards, n_scrambles, heuristic):
    """
    solves n_boards (scrambled n_scrambles times) and then returns each of the board states along the way to the
    solution accompanied by the remaining number of steps in that solution. this is because we want the
    neural network to map the board state onto the number of steps remaining in the solution (the heuristic function).
    :param n_boards: number of board states to solve
    :param n_scrambles: number of times to scramble the boards
    :param heuristic: the heuristic to find the solution
    :return: numpy array of states and a numpy array of their corresponding remaining steps
    """

    states = []
    values = []

    boards = []
    for _ in range(n_boards):
        this_board = Board(np.array([[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 0]]))
        this_board.scramble(n_scrambles)
        boards.append(this_board)

    for board in boards:
        solution,_ = a_star(board, heuristic)
        solution = solution[:-1]
        length = len(solution)
        for i,state in enumerate(solution):
            states.append(state.get_board())
            values.append(length - i)

    for i,state in enumerate(states):
        states[i] = transform(state)

    return np.array(states), np.array(values)


def neural_heuristic(board):
    return model.predict(transform(board.get_board()).reshape((1,256)))


def train(max_scrambles, nn_dim_string):
    complete_x = []
    complete_y = []
    for i in range(1, max_scrambles+1):
        t0 = time.time()
        x_train, y_train = training_data(200, i, neural_heuristic)
        for x in x_train:
            complete_x.append(x)
        for y in y_train:
            complete_y.append(y)
        model.fit(np.array(complete_x), np.array(complete_y), epochs=25)
        if i % 5 == 0:
            model.save("models/neural_network - {} - {}.h5".format(nn_dim_string, i))
        print(i, "iterations completed out of", max_scrambles)
        print("iteration time:", time.time() - t0)



if __name__ == "__main__":

    train(35, "512x1024x512")