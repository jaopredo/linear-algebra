import numpy as np

def householder_qr(A):
    A = A.copy().astype(float)
    m, n = A.shape
    Q = np.eye(m)

    for k in range(n):
        # Extrai o vetor da coluna atual
        x = A[k:, k]
        e = np.zeros_like(x)
        e[0] = np.linalg.norm(x) * (-1 if x[0] < 0 else 1)
        u = x + e
        v = u / np.linalg.norm(u)

        # Construir a matriz de Householder
        H_k = np.eye(m)
        H_k[k:, k:] -= 2.0 * np.outer(v, v)

        A = H_k @ A
        Q = Q @ H_k.T  # Q acumula as transpostas

    Q_reduced = Q[:, :n]
    R_reduced = A[:n, :]

    return Q_reduced, R_reduced


def modified_gram_schmidt(A):
    A = A.astype(float)  # Garante que estamos trabalhando com floats
    m, n = A.shape
    Q = np.zeros((m, n))
    R = np.zeros((n, n))
    
    for k in range(n):
        v = A[:, k].copy()
        for j in range(k):
            R[j, k] = np.dot(Q[:, j], v)
            v = v - R[j, k] * Q[:, j]
        R[k, k] = np.linalg.norm(v)
        if R[k, k] == 0:
            raise ValueError("Colunas linearmente dependentes detectadas.")
        Q[:, k] = v / R[k, k]
    
    return Q, R
