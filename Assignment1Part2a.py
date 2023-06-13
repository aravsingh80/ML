import sys
import ast
import math
import random
import sympy as sp
import numpy as np
x = float(sys.argv[1])
y = float(sys.argv[2])
# def firstFunct(x, y): return (x*(y**3))-(x**5)+(3*y*y)-3
# def derivFirstFunctX(x, y): return (y**3)-(5*(x**4))
# def derivFirstFunctY(x, y): return (3*x*y*y)+(6*y)
# def secondDerivFirstFunctXX(x, y): return (20*(x**3))
# def secondDerivFirstFunctYY(x, y): return (6*x*y)+6
# def secondDerivFirstFunctXY(x, y): return 3*y*y

# def firstFunct(x, y): return (x**3)-(y**2)+7
# def derivFirstFunctX(x, y): return 3*x*x
# def derivFirstFunctY(x, y): return -2*y
# def secondDerivFirstFunctXX(x, y): return 6*x
# def secondDerivFirstFunctYY(x, y): return -2
# def secondDerivFirstFunctXY(x, y): return 0

def firstFunct(x, y): return float((x**4)+(y**2)+(3*x))
def derivFirstFunctX(x, y): return float((4*x*x*x)+3)
def derivFirstFunctY(x, y): return float(2*y)
def secondDerivFirstFunctXX(x, y): return float(12*x*x)
def secondDerivFirstFunctYY(x, y): return 2.0
def secondDerivFirstFunctXY(x, y): return 0.0
def firstFunct2(x, y): return (y**3)+(x**2)+(x*y)-y
def derivFirstFunctX2(x, y): return (2*x)+y
def derivFirstFunctY2(x, y): return (3*y*y)+x-1
def secondDerivFirstFunctXX2(x, y): return 2.0
def secondDerivFirstFunctYY2(x, y): return 6*y
def secondDerivFirstFunctXY2(x, y): return 1.0

def localMin(f, dfx, dfy, dfxx, dfyy, dfxy, x, y):
  root = np.array([[x],[y]])
  hessian = np.array([[dfxx(x, y), dfxy(x, y)], [dfxy(x, y), dfyy(x, y)]])
  oldCheck = True
  #print((abs(dfx(root[0][0], root[1][0]))) < 0.00001 )
  # f(x) + (df(x)*(x-root))
  #if int(dfx(root[0][0], root[1][0])) == 0 and int(dfy(root[0][0], root[1][0])) == 0: return f(root[0][0], root[1][0])
  while ((abs(dfx(root[0][0], root[1][0]))) > 0.00001 or abs((dfy(root[0][0], root[1][0]))) > 0.00001 ): 
    # print(oldRoot)
    # print(root)
    # print(f(x) + (df(x)*(x-root)))
    root = root - (np.linalg.inv(hessian)@np.array([[dfx(root[0][0], root[1][0])],[dfy(root[0][0], root[1][0])]]))
    # print(root)
    #print(round(root[1][0], 3) == round(oldRoot[1][0], 3))
    # if round(root[0][0], 3) == round(oldRoot[0][0], 3) and round(root[1][0], 3) == round(oldRoot[1][0], 3): oldCheck = False
    # print(oldCheck)
    # print((dfx(root[0][0], root[1][0])) != 0 or (dfy(root[0][0], root[1][0])) != 0)
  return (root, f(root[0][0], root[1][0]))
print(localMin(firstFunct, derivFirstFunctX, derivFirstFunctY, secondDerivFirstFunctXX, secondDerivFirstFunctYY, secondDerivFirstFunctXY, x, y))
print(localMin(firstFunct2, derivFirstFunctX2, derivFirstFunctY2, secondDerivFirstFunctXX2, secondDerivFirstFunctYY2, secondDerivFirstFunctXY2, x, y))