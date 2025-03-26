import numpy as np

def generate_invertible_matrix(n: int = 2) -> np.ndarray:
    """Generates a invertible squared nxn matrix

    Args:
        n (int, optional): How many lines and columns the matrix have. Defaults to 2.

    Returns:
        np.ndarray: Returns the invertible nxn matrix
    """
    while True:
        A = np.random.randint(-10, 10, (n, n))
        if np.linalg.det(A) != 0:
            return A


def generate_column_vector(lines: int = 2) -> np.ndarray:
    """Generates a random column vector

    Args:
        lines (int, optional): How many lines the column vector will have. Defaults to 2.

    Returns:
        np.ndarray: The generated column array
    """
    return np.random.randint(-10, 10, (lines, 1))


def generate_strictly_diagonal_dominant_matrix(n: int, min_val: int = 1, max_val: int = 10) -> np.ndarray:
    """Generates a squared n x n strictly diagonal dominant

    Args:
        n (int): Dimension of the matrix
        min_val (int, optional): Minimum value for the elements outside the diagonal. Defaults to 1.
        max_val (int, optional): Maximum value for the elements outside the diagonal. Defaults to 10.

    Returns:
        np.ndarray: Matrix strictly diagonal dominant
    """
    A = np.random.randint(min_val, max_val, (n, n))

    # Adjusting the diagonal matrix
    for i in range(n):
        A[i, i] = np.sum(np.abs(A[i])) + 1  # Garante a condição de dominância

    return A
