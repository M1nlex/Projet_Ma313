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



def Cholesky(A):
    """
    Fonction qui calcule L la matrice de la décomposition de
    Cholesky de A une matrice réelle symétrique définie positive
    (A=LL^T où L est triangulaire inférieure).
    La fonction ne vérifie pas que A est symétrique.
    La fonction rend L.
    """

    n, m = A.shape
    if n != m:
        raise Exception('Matrice non carrée')
    L = np.zeros((n , n))
    for i in range(n):
        s = 0.
        for j in range(i):
            s = s+L[i , j]**2
        R = A[i, i]-s
        if R <= 0:
            raise Exception('Matrice non définie positive')
        L[i, i] = np.sqrt(R)
        for j in range(i+1, n):
            s = 0.
            for k in range(i):
                s = s+L[i, k]*L[j, k]
            L[j, i] = (A[j, i]-s)/L[i, i]
    return L