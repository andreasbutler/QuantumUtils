import numpy as np
from scipy import linalg as la

"""Linear algebra utilities."""

def svd_inverse(matrix):
    """Return the inverse of a matrix computed from its SVD. This is useful
    in situations in which the inverse of an ill-conditioned matrix is
    needed."""
    U, S, V = np.linalg.svd(matrix)

    dim = S.shape[0]
    S = la.diagsvd(S, dim, dim)
    V = np.matrix(V)
    U = np.matrix(U)

    # Compute the inverse SVD
    V_dag_S = np.dot(V.getH(), np.linalg.inv(S))
    V_dag_S_U_dag = np.dot(V_dag_S, U.getH())

    return V_dag_S_U_dag
