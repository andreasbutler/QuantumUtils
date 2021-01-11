import numpy as np
import qutip as qutip

from matplotlib import pyplot as plt

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
