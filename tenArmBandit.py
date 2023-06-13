import random
def bandit(testNum, armIdx, pullVal):
    if testNum == 0:
        testNum = random.randint(1, 1000)
        armIdx = 10
        pullVal = 0
    elif testNum > 0: 
        armIdx = armIdx
        pullVal = pullVal
    return armIdx

import math
def bandit(testNum, armIdx, pullVal):
    global count, total
    if testNum == 0:
        count = [0 for x in range(10)]
        total = [0 for x in range(10)]
        return 0
    if testNum > 0: 
        count[armIdx] += 1
        total[armIdx] += pullVal
    final = []
    for x in range(10):
        mean = total[x]/count[x]
        temp = 0.8 * math.sqrt(math.log(testNum)/count[x])
        final.append(mean + temp)
    return final.index(max(final))

import random

def bandit(testNum, armIdx, pullVal):
    global armcounts, totals
    if testNum == 0:
        totals = [0.0]*armIdx
        armcounts = [0]*armIdx
        epsilon = 0.05
        return 0
    else:
        armcounts[armIdx] += 1
        totals[armIdx] += (pullVal - totals[armIdx])/armcounts[armIdx]
        epsilon = 1/(testNum ** 0.5) 
        if random.random() < epsilon:
            nextArm = random.randint(0, len(totals) - 1)
        else:
            nextArm = max(range(len(totals)), key=totals.__getitem__)
        return nextArm
  # Bandit pull maximizer; if testNum ...
  # ==0 => new bandit initialization; armIdx contains the
  #         number of arms (10); pullVal not used
  # > 0 => testNum contains the pull number (from 1 to 999)
  #        armIdx has the index of the arm that was requested
  #            to be pulled in the prior call
  #        pullVal has the value that resulted from the pull
  # The return val is always the idx of the next
  #     arm to pull in [0,# of arms)


#from numpy.random import normal; from karmbandit import *

# def tester():
#     res = []
#     for i in range(1000):
#         bandits = [k for k in normal(0, 1, 10)]
#         reward = 0
#         armpl = bandit(0, 10, None)
#         for k in range(1, 1001):
#             reward += (val:=normal(bandits[armpl], 1))
#             armpl = bandit(k, armpl, val)
#         res.append(reward/k)
#         if not (i+1) % 10: print(round(sum(res[i-9:i+1])/10, 2), end=" ")
#         if not (i+1) % 100: print()
#     print(f"SCORE: {sum(res)/(len(res)/1000)}") 