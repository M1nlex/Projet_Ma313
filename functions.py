import numpy as np
import function_base as fct
import matplotlib.pyplot as plt


def ResolMCEN(A, b):
    aa = A.T@A
    bb = A.T@b
    #return np.linalg.solve(aa,bb)

    l = fct.Cholesky(aa)
    lt = l.T

    y = fct.ResolTriInf(l, bb)
    x = fct.ResolTriSup(lt, y)

    return x




def ResolMCQr(A,b):
    pass

def ResolMCNP(A,b):
    pass



