import sys
import ast
import math
import random
import sympy as sp
import numpy as np
x = int(sys.argv[1])
def firstFunct(x): return (x*x*7)-(29*x)+1
def derivFirstFunct(x): return (14*x)-29
def rootF(f, df, x):
  root = x
  # f(x) + (df(x)*(x-root))
  while f(x) + (df(x)*(x-root)) != 0: 
    # print(f(x) + (df(x)*(x-root)))
    print(root)
    root = root - (f(root)/df(root))
  return root
#error: infinite while loop