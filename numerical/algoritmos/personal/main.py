import numpy as np
import matplotlib.pyplot as plt
from qr import householder_qr, modified_gram_schmidt
import utils

np.random.seed(0)

m = 100
n = 15
t = np.linspace(0, 1, m)
A = np.vander(t, n, True)
b = np.exp(np.sin(4*t))/2.00678728e+03

Q, R = householder_qr(A)
x = np.linalg.solve(R, Q.T @ b)
print(1-x[-1])

b = b.reshape(-1, 1)

Q, R = householder_qr(np.c_[A, b])
Qb = R[0:n, n]
R = R[0:n, 0:n]
x = np.linalg.solve(R, Qb)
print(1-x[-1])

Q, R = modified_gram_schmidt(A)
x = np.linalg.solve(R, Q.T @ b)
print(1-x[-1])

Q, R = modified_gram_schmidt(np.c_[A, b])
Qb = R[0:n, n]
R = R[0:n, 0:n]
x = np.linalg.solve(R, Qb)
print(1-x[-1])

x = np.linalg.solve(A.T @ A, A.T @ b)
print(1-x[-1])

U, S, Vt = np.linalg.svd(A, full_matrices=False)
x = (Vt.T * 1/S) @ (U.T @ b)
print(1-x[-1])
