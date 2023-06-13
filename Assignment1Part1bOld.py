import sys
import ast
import math
import random
import sympy as sp
import numpy as np
x = int(sys.argv[1])
def firstFunct(x): return (x*x*7)-(29*x)+1
def derivFirstFunct(x): return (14*x)-29
def secondFPrime(x): return (36*(x**6))+(15*(x**2))-7
def secondF(x): return (6*(x**7))+(5*(x**3))-(7*x)+3
def secondFDoublePrime(x): return (216*(x**5))+(30*x)
def localMin(f, df, df2, x):
  root = x
  # f(x) + (df(x)*(x-root))
  while f(x) + (df(x)*(x-root)) != 0: 
    # print(f(x) + (df(x)*(x-root)))
    root = root - (df(root)/df2(root))
  return f(root)
print(localMin(secondF, secondFPrime, secondFDoublePrime, x))
#error: infinite loop