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

def firstFunct(x, y): return math.exp(x)+y
def firstFunct2(x, y): return x-1
def derivFirstFunctX(x, y): return math.exp(x)
def derivFirstFunctY(x, y): return 1
def secondDerivFirstFunctX(x, y): return 1
def secondDerivFirstFunctY(x, y): return 0

def firstFunct23(x, y): return (x**2)+(y**2)-4
def firstFunct22(x, y): return (4*x*x)-(y*y)-4
def derivFirstFunctX2(x, y): return 2*x
def derivFirstFunctY2(x, y): return 2*y
def secondDerivFirstFunctX2(x, y): return 8*x
def secondDerivFirstFunctY2(x, y): return -2*y

def firstFunct24(x, y): return (x**2)+(y**2)-4
def firstFunct25(x, y): return (4*x*x)-(y*y)-4
def derivFirstFunctX23(x, y): return 2*x
def derivFirstFunctY23(x, y): return 2*y
def secondDerivFirstFunctX23(x, y): return 8*x
def secondDerivFirstFunctY23(x, y): return -2*y



def localMin(f, f2, dfx, dfy, df2x, df2y, x, y):
  root = np.array([[x], [y]])
  jacobian = np.array([[dfx(x, y), dfy(x, y)], [df2x(x, y), df2y(x, y)]])
  fa = np.array([[f(x, y)],[f2(x, y)]])
  x, y = root[0][0], root[1][0]
  while ((abs(f(x, y))) > 0.000000001 or abs((f2(x, y))) > 0.000000001): 
    print(f(x, y))
    root = root - np.linalg.inv(jacobian)@fa
    x = root[0][0]
    y = root[1][0]
    jacobian = np.array([[dfx(x, y), dfy(x, y)], [df2x(x, y), df2y(x, y)]])
    fa = np.array([[f(x, y)],[f2(x, y)]])
  return (root, f(x, y))

# def localMin(f, f2, dfx, dfy, df2x, df2y, x, y):
#   root = np.array([[x], [y]])
#   jacobian = np.array([[dfx(x, y), dfy(x, y)], [df2x(x, y), df2y(x, y)]])
#   fa = np.array([[f(x, y)],[f2(x, y)]])
#   x, y = root[0][0], root[1][0]
#   while ((abs(f(x, y))) > 0.000000001 or abs((f2(x, y))) > 0.000000001): 
#     print(f(x, y))
#     root = root - (np.linalg.inv(jacobian)@np.array([[f(x, y)],[f2(x, y)]]))
#     x, y = root[0][0], root[1][0]
#     jacobian = np.array([[dfx(x, y), dfy(x, y)], [df2x(x, y), df2y(x, y)]])
#   return [(f(x, y)), (f2(x, y))]
print(localMin(firstFunct, firstFunct2, derivFirstFunctX, derivFirstFunctY, secondDerivFirstFunctX, secondDerivFirstFunctY , x, y))
print(localMin(firstFunct23, firstFunct22, derivFirstFunctX2, derivFirstFunctY2, secondDerivFirstFunctX2, secondDerivFirstFunctY2 , x, y))