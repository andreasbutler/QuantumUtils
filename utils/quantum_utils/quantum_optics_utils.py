import qutip as qutip
import numpy as np

"""A utility file for quantum optics manipulations."""

def compute_matrix_of_moments(state, max_n):
    """Compute the matrix of normally ordered moments given by <a_dag^n a^m>"""
    dim = state.shape[0]
    a = qutip.destroy(dim)
    a_dag = a.dag()
    
    moments = np.zeros((max_n, max_n), dtype=np.complex)

    for n in range(max_n):
        for m in range(max_n):
            a_dag_n_a_m = pow(a_dag, n) * pow(a, m)
            moments[n][m] = qutip.expect(a_dag_n_a_m, state)

    return moments


def generate_joint_2_photon_quadrature_histogram(rho, dim, coherent_state_povm):
    """Computes the 2 photon generalization of the Husimi Q function.

    Assumes that both photon states should have the same resolution and
    centering in quadrature space and so uses a single quadrature mesh to 
    compute the 4D histogram of joint 2-photon quadrature value amplitudes
    given by Q_abcd = <a + ib, c + id|rho|a + ib, c + id>.

    Args:
        rho: The state whose 4D quadrature histogram is being computed
        dim: The dimension of each single-mode truncated Hilbert space
        coherent_state_povm: The mesh of complex values over which the
            quadrature histogram is defined
    Returns:
        A 4D numpy array representing the joint 2 photon quadrature
        histogram"""
    number_of_single_mode_quadrature_buckets = len(coherent_state_povm)
    quadrature_histogram = np.zeros((number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets))
    for i, p_rowA in enumerate(coherent_state_povm):
        for j, coherent_state_A in enumerate(p_rowA):
            for k, p_rowB in enumerate(coherent_state_povm):
                for l, coherent_state_B in enumerate(p_rowB):
                    joint_state = qutip.tensor(coherent_state_A,
                                               coherent_state_B)
                    quadrature_histogram[i][j][k][l] = qutip.expect(rho,
                                                                joint_state)
    return quadrature_histogram


def generate_joint_3_photon_quadrature_histogram(rho, dim, coherent_state_povm):
    """Computes the 2 photon generalization of the Husimi Q function.

    Assumes that both photon states should have the same resolution and
    centering in quadrature space and so uses a single quadrature mesh to 
    compute the 4D histogram of joint 2-photon quadrature value amplitudes
    given by Q_abcd = <a + ib, c + id|rho|a + ib, c + id>.

    Args:
        rho: The state whose 4D quadrature histogram is being computed
        dim: The dimension of each single-mode truncated Hilbert space
        coherent_state_povm: The mesh of complex values over which the
            quadrature histogram is defined
    Returns:
        A 4D numpy array representing the joint 2 photon quadrature
        histogram"""
    number_of_single_mode_quadrature_buckets = len(coherent_state_povm)
    quadrature_histogram = np.zeros((number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets,
                                     number_of_single_mode_quadrature_buckets))
    for i, p_rowA in enumerate(coherent_state_povm):
        for j, coherent_state_A in enumerate(p_rowA):
            for k, p_rowB in enumerate(coherent_state_povm):
                for l, coherent_state_B in enumerate(p_rowB):
                    for m, p_rowC in enumerate(coherent_state_povm):
                        for n, coherent_state_C in enumerate(p_rowC):
                            joint_state = qutip.tensor(coherent_state_A,
                                                       coherent_state_B,
                                                       coherent_state_C)
                            quadrature_histogram[i][j][k][l][m][n] = \
                                qutip.expect(rho, joint_state)
    return quadrature_histogram
