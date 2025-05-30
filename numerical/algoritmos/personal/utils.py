import numpy as np


def vandermonde_matrix(points: np.ndarray, degrees: int) -> np.ndarray:
    """Function that generates the PHI matrix (Used to make the linear regression). With the exception that the
    generated matrix here doesn't contains the first column as x⁰ because it will be used later in a function wich
    adds the 1's column automatically

    Args:
        points (np.ndarray): Vector containing the X points
        degrees (int): The number of columns the PHI matrix will have

    Returns:
        np.ndarray: The PHI matrix
    """
    phi = np.zeros((points.shape[0], degrees))

    for i, xi in enumerate(points):
        for j in range(degrees):
            phi[i][j] = xi**(j+1)
    
    return phi


def apply_polynomial_to_space(sample: np.ndarray, coefficients: np.ndarray) -> np.array:
    """Function that receives a unidimensional Array (column array or common array), applies a polynomial function
    to it and returns the vector containing the results of the applied polynomial

    Args:
        sample (np.ndarray): Array containting the points that will be applied
        coefficients (np.ndarray): Coefficients of the polynomial in terms that each i entry will be applied in x^i

    Returns:
        np.ndarray: The vector containing each entry of the "sample" vector applied to the polynomial function
    """
    y_vector = np.zeros(sample.shape)

    for i, ci in enumerate(coefficients):
        y_vector += ci*(sample**i)
    
    return y_vector