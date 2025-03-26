"""Module containing utility functions
"""
import numpy as np


def is_strictly_diagonal_dominant(Matrix: np.ndarray) -> bool:
    """Checks if a squared matrix is strictily diagonal diminant

    Args:
        Matrix (np.ndarray): The squared matrix to check

    Raises:
        ValueError: Raises a Value error if the matrix passed is not square

    Returns:
        bool: Returns if the matrix is strictly dominant
    """
    sum_vector = np.absolute(Matrix).sum(axis=1, )
    diag_vector = np.diag(Matrix)

    return (diag_vector > (sum_vector - diag_vector)).all()
