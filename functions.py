import numpy as np
import matplotlib.pyplot as plt

def ResolMCEN(A,b):
    aa = A.T@A
    bb = A.T@b
    return np.linalg.solve(aa,bb)

def ResolMCQr(A,b):
    pass

def ResolMCNP(A,b):
    pass
