import numpy as np
from functions import *

# Exercice 1 :
a1 = np.array([[1, 2], [2, 3], [-1, 2]])
b1 = np.array([[12], [17], [6]])
# print(ResolMCEN(a1, b1))
# print(ResolMCNP(a1, b1))
# print(ResolMCQr(a1, b1))

# Exercice 2 :
a2 = np.array([[1, 21], [-1, -5], [1, 17], [1, 17]])
b2 = np.array([[3], [-1], [1], [1]])
# print(ResolMCEN(a2, b2))
# print(ResolMCNP(a2, b2))
# print(ResolMCQr(a2, b2))

# Exercie 3 :
x3 = np.array([0.3, -2.7, -1.9, 1.2, -2.6, 2.7, 2.0, -1.6, -0.5, -2.4])
y3 = np.array([2.8, -9.4, -4.5, 3.8, -8.0, 3.0, 3.9, -3.5, 1.3, -7.6])
'''
plt.plot(x3, y3, 'x')
n = x3.size
A = np.ones((n, 3))

A[:, 1] = x3
A[:, 2] = x3**2

U1 = ResolMCEN(A, y3)
U2 = ResolMCNP(A, y3)
U3 = ResolMCQr(A, y3)


alpha1, beta1, gamma1 = U1[0]
alpha2, beta2, gamma2 = U1[0]
alpha3, beta3, gamma3 = U1[0]
x_min = np.min(x3)
x_max = np.max(x3)
X = np.linspace(x_min, x_max)
Y1 = alpha1+beta1*X+gamma1*X**2
Y2 = alpha2+beta2*X+gamma2*X**2
Y3 = alpha3+beta3*X+gamma3*X**2

plt.plot(X, Y1, '-r', label='ResolMCEN')
plt.plot(X, Y2, '-b', label='ResolMCNP')
plt.plot(X, Y3, '-y', label='ResolMCQr')
plt.legend()
plt.show()
'''


# test_minimum(a1, b1)
cercle()
