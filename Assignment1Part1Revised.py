import sys
import ast
import math
import random
import sympy as sp
import numpy as np
from math import pi , acos , sin , cos
x = float(sys.argv[1])
def firstFunct(x): return (x*x*x)-(2*x)
def derivFirstFunct(x): return (3*x*x)-2
def secondF(x): return (2*x*x*x*x)-(4*x*x)+1
def secondFPrime(x): return (8*x*x*x)-(8*x)
def thirdFPrime(x): return cos(x)
def thirdF(x): return sin(x)
def fourthF(x): return (1/(1+(math.exp(-x)))) -0.75
def fourthFPrime(x): return math.exp(x)/((math.exp(x)+1)**2)
def rootF(f, df, x):
  root = x
  # f(x) + (df(x)*(x-root))
  while abs(f(root)) > 0.000000001: 
  #while (f(root))!=0: 
    # print(f(x) + (df(x)*(x-root)))
    #print(root)
    root = root - (f(root)/df(root))
  return (root, f(root))
print(rootF(firstFunct, derivFirstFunct, x))
print(rootF(secondF, secondFPrime, x))
print(rootF(thirdF, thirdFPrime, x))
print(rootF(fourthF, fourthFPrime, x))
