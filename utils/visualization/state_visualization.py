import numpy as np
import qutip as qutip

from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D

"""A utility file for visualizing states."""


def plot_fock_basis_probabilities(state):
    """Plots the probabilities of measuring `state` in each of the Fock states
    from |0> to |`state`.dim>."""
    dim = state.shape[0]
    basis = [qutip.basis(dim, i) * qutip.basis(dim, i).dag() for i in range(dim)]

    coherent_coeffs = [qutip.expect(fock_n, state) for fock_n in basis]
    coherent_probabilities = [np.abs(coeff) for coeff in coherent_coeffs]
    
    fig, ax = plt.subplots(1)
    ax.bar(range(dim), coherent_probabilities)
    ax.set_xlabel('Fock state index')
    ax.set_ylabel('Probability')


def plot_fock_basis_amplitudes(state):
    """Plots the probabilities of measuring `state` in each of the Fock states
    from |0> to |`state`.dim>."""
    dim = state.shape[0]
    basis = [qutip.basis(dim, i) * qutip.basis(dim, i).dag() for i in range(dim)]

    coherent_coeffs = [qutip.expect(fock_n, state) for fock_n in basis]
    coherent_probabilities = [np.abs(coeff)**(1/2) for coeff in coherent_coeffs]
    
    fig, ax = plt.subplots(1)
    ax.bar(range(dim), coherent_probabilities)
    ax.set_xlabel('Fock state index')
    ax.set_ylabel('Amplitude')


def plot_fock_basis_amplitudes_with_reference(
        state,
        ref_state,
        legend_labels,
        title=None):
    """Plots the probabilities of measuring `state` in each of the Fock states
    from |0> to |`state`.dim> along with a reference state."""
    dim = state.shape[0]
    basis = [qutip.basis(dim, i) * qutip.basis(dim, i).dag() for i in range(dim)]

    coherent_coeffs = [qutip.expect(fock_n, state) for fock_n in basis]
    coherent_amplitudes = [np.abs(coeff)**(1/2) for coeff in coherent_coeffs]
    reference_coeffs = [qutip.expect(fock_n, ref_state) for fock_n in basis]
    reference_amplitudes = [np.abs(coeff)**(1/2) for coeff in reference_coeffs]
    
    fig, ax = plt.subplots(1)
    ax.bar(range(dim), reference_amplitudes, label=legend_labels['ref'], zorder=0)
    ax.scatter(range(dim), coherent_amplitudes, label=legend_labels['data'], zorder=1)
    ax.legend()
    ax.set_xlabel('Fock state index')
    ax.set_ylabel('Amplitude')
    
    if title is not None:
        ax.set_title(title)


def windowed_density_matrix(rho, n, m):
    """Returns the submatrix rho_ij for n <= i,j <= m."""
    return rho[n:m + 1, n:m + 1]


def plot_density_matrix(rho):
    """Plots a density matrix."""
    fig = plt.figure( figsize=(8,5))
    ax = Axes3D(fig, azim=-35, elev=35)
    qutip.matrix_histogram_complex(qutip.Qobj(rho).unit(), fig=fig, ax=ax)
    return fig, ax


def plot_image(im, xs, ps):
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pcolormesh(xs, ps, im)
    return fig, ax


def plot_qfunc(rho, xs, ps):
    qfunc = qutip.qfunc(rho, xs, ps, g=2)
    fig, ax = plot_image(qfunc, xs, ps)

    return fig, ax

