import sys
import ast
import math
import random
import sympy as sp
import numpy as np
from math import sin, cos
x = float(sys.argv[1])
def firstFunct(x): return (x*x*x)-(2*x)
def derivFirstFunct(x): return (3*x*x)-2
def secDerivFirst(x): return 6*x
def secondF(x): return (2*x*x*x*x)-(4*x*x)+1
def secondFPrime(x): return (8*x*x*x)-(8*x)
def secDerivSec(x): return (24*x*x)-8
def thirdFPrime(x): return cos(x)
def thirdF(x): return sin(x)
def secDerivThird(x): return -sin(x)
def fourthF(x): return (1/(1+(math.exp(-x))))-0.75
def fourthFPrime(x): return math.exp(x)/((math.exp(x)+1)**2)
def secDerivFourth(x): return (-(math.exp(x)-1)*math.exp(-2*x))/(math.exp(-x)+1)**3
def localMin(f, df, df2, x):
  root = x
  # f(x) + (df(x)*(x-root))
  while abs(df(root)) > 0.000000001: 
    # print(f(x) + (df(x)*(x-root)))
    root = root - (df(root)/df2(root))
  return (root, f(root))
print(localMin(firstFunct, derivFirstFunct, secDerivFirst, x))
print(localMin(secondF, secondFPrime, secDerivSec, x))
print(localMin(thirdF, thirdFPrime, secDerivThird,x))
#print(secDerivFourth(0))
print(localMin(fourthF, fourthFPrime, secDerivFourth, x))