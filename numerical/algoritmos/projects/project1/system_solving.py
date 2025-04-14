"""This module contains functions for solving linear systems of the form Ax = b
"""
import numpy as np


def jacobi(A: np.ndarray, b: np.ndarray, n: int = 100) -> np.ndarray:
    """This function uses the Jacobi method for computing the solution of the linear system Ax = b

    Args:
        A (np.ndarray): Matrix A
        b (np.ndarray): Vector b
        n (int, optional): The amout of times the initial guess will be iterated to converge to the solution. Default to 100

    Returns:
        np.ndarray: Returns the approximation to the solution of the system
    """
    x = np.zeros()
    

def gauss_seidel():
    ...
