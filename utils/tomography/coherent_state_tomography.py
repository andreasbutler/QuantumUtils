import numpy as np
import qutip as qutip
import time as time

from matplotlib import pyplot as plt
from scipy import ndimage

import sys
import os
sys.path.append(os.path.abspath('../'))
from math_utils import linear_algebra_utils as linalg_utils

"""Utilities for performing photonic coherent state MLE tomography."""

def generate_complex_mesh(max_x, mesh_size):
    """Makes a square mesh of complex numbers centered at the origin.

    The mesh is returned as well as the one dimensional arrays of the
    x and y coordinates of the mesh (x and p are used because of the
    physics naming convention for phase space variables) because these
    1D arrays are useful to have for plotting.

    Returns:
        alphas: A 2D mesh of complex values.
        xs: The x coordinates of the mesh.
        ps: The y coordinates of the mesh."""
    xs = np.arange(-max_x, max_x + mesh_size, step=mesh_size)
    ps = np.arange(-max_x, max_x + mesh_size, step=mesh_size)
    x_grid, p_grid = np.meshgrid(xs, ps)
    alphas = x_grid + 1j*p_grid

    return alphas, xs, ps
    

def generate_coherent_state_POVM(max_x, mesh_size, dimension, noise_base=None):
    """Generates a square mesh of coherent state POVM elements centered at the
    origin. Because coherent states are overcomplete, any state can be written
    diagonally in the coherent state representation.

    If a noise state is supplied via noise_base, the POVM is modified so that
    each coherent state in the noiseless POVM inherits the fluctuation
    statistics of the noisy state (read: the POVM mesh is instead the noise
    state displaced to every point of the specified lattice in phase space.)"""
    alpha_mesh, xs, ps = generate_complex_mesh(max_x, mesh_size)

    # If no specific noise state is supplied, use the vaccuum state as the
    # base of the coherent state mesh (because ideal coherent states are just
    # displaced vaccuum states).
    if noise_base is None:
        noise_base = qutip.coherent_dm(dimension, 0)

    noisy_POVM = []
    for row in alpha_mesh:
        noisy_POVM_row = []
        for alpha in row:
            noisy_POVM_element = 1/np.pi * qutip.displace(dimension, alpha) \
                                     * noise_base \
                                     * qutip.displace(dimension, alpha).dag()
            noisy_POVM_row.append(noisy_POVM_element)
        noisy_POVM.append(noisy_POVM_row)
    
    return noisy_POVM, alpha_mesh, xs, ps


def sum_povm_elements(povm):
    """Sums over povm elements and returns sum"""
    dim = povm[0][0].shape[0]
    povm_sum = qutip.Qobj(np.zeros((dim, dim)))

    for row in povm:
        for povm_element in row:
            povm_sum += povm_element

    return povm_sum


def extract_G_inv_from_POVM(povm):
    """Returns the inverse of the sum of the povm elements."""
    povm_sum = sum_povm_elements(povm)
    G_inv = qutip.Qobj(linalg_utils.svd_inverse(povm_sum))

    return G_inv


def evaluate_Q_function(input_state, coherent_state_POVM):
    """Evaluates the Q function for a (possibly noisy) coherent state POVM"""
    Q_vals = [[qutip.expect(coherent_state_povm_element, input_state)
                   for coherent_state_povm_element in row]
                       for row in coherent_state_POVM]
    return Q_vals


def evaluate_thermally_noisy_Q_function(input_state, noise_photons, xs, ps):
    """Turn an ideal state's Q function into the Q function of the ideal state
    affected by thermal noise by convolving the ideal Q function with the
    thermal state's P function."""
    ideal_qfunc = qutip.qfunc(input_state, xs, ps, g=2)
    mesh_size = xs[1] - xs[0]
    convolution_sigma = (noise_photons / 2)**(1/2) / (mesh_size)
    noisy_Q_function = ndimage.gaussian_filter(ideal_qfunc,
                                               convolution_sigma,
                                               mode='nearest')
    return noisy_Q_function


def MLE_evaluate_R(state,
                   POVM_mesh, 
                   measured_POVM_frequencies,
                   frequency_threshold=0):
    """Evaluates the iterative state update operator."""
    R = qutip.Qobj(np.zeros(state.shape))
    for i, row in enumerate(POVM_mesh):
        for j, POVM_ij in enumerate(row):
            p_ij = qutip.expect(POVM_ij, state)
            f_ij = measured_POVM_frequencies[i][j]
            if np.abs(p_ij) > frequency_threshold:
                R += POVM_ij * f_ij / p_ij
    return R


def MLE_update_state_estimate(current_state, 
                              R, 
                              G_inv,
                              identity_mixin=0):
    """Updates the MLE best estimate state."""
    updated_state = G_inv * R * current_state * R * G_inv
    # Explicitly make the returned state Hermitian. This is necessary becase of
    # possible numerical floating point accuracy issues with the G_inv matrix 
    # which is calculated from a numpy.linalg.inv call.
    updated_state = 1/2 * (updated_state + updated_state.dag())

    # If we must regularize update, do so
    if identity_mixin > 0:
        print('oooga')
        dim = current_state.shape[0]
        identity = dim * qutip.maximally_mixed_dm(dim)
        updated_state = identity_mixin * identity * current_state \
                         + (1 - identity_mixin) * updated_state.unit()

    return updated_state.unit()


def perform_coherent_state_MLE(povm,
                               measured_POVM_frequencies, 
                               number_of_iterations, 
                               rho0=None, 
                               rho_ideal=None,
                               identity_mixin=0,
                               frequency_threshold=0):
    """Performs coherent state MLE given a POVM and the correspondinding
    measured POVM frequencies.

    In reality this function is good for a fairly general MLE reconstruction,
    allowing for potentially noncomplete POVM elements. It is not specific to
    bosonic spaces or any sort of POVM structure actually.

    Args:
        povm: The POVM operators from the tomography.
        measured_POVM_frequencies: The frequency with which the corresponding
            (by index) POVM elements were measured.
        number_of_iterations: The number of MLE iterations.
        rho0: The starting point for the tomographic reconstruction algorithm
        rho_ideal: The ideal recovered state from tomography. If specified, the
            intermediate fidelities of the reconstructed state after each MLE
            iteration will be returned.
        identity_mixin: The fraction of the identity to be mixed into the state
            update operator during each iteration. Is supposedly necessary to
            regulate the convergence of the algorithm, but I've never needed it
            to be greater than 0.
        frequency_threshold: A threshold below which POVM elements will be
            excluded during the construction of the state iteration operator to
            prevent floating point errors for POVMs with very small overlap.
            I've never needed this."""
    # Extract the POVM completion operator
    G_inv = extract_G_inv_from_POVM(povm)
    
    # Initialize unbiased mixed state as MLE initial state
    dim = povm[0][0].shape[0] # ugly way to extract dimension
    state = 1/dim * qutip.qeye(dim)
    
    if rho0 is not None:
        state = rho0
    
    intermediate_fidelities = []
    
    # Run MLE
    print(number_of_iterations)
    for i in range(number_of_iterations):
        R = MLE_evaluate_R(state, 
                           povm, 
                           measured_POVM_frequencies,
                           frequency_threshold=frequency_threshold)
        state = MLE_update_state_estimate(state, 
                                          R, 
                                          G_inv, 
                                          identity_mixin=identity_mixin)
        if rho_ideal is not None:
            intermediate_fidelities.append(qutip.fidelity(state, rho_ideal))
    
    if rho_ideal is not None:
        return state, intermediate_fidelities
    
    return state


### PLOTTING UTILITIES #################
def get_plotting_limits(data_sets):
    vmin = np.min(data_sets[0])
    vmax = np.max(data_sets[0])
    for data_set in data_sets:
        vmin = min(vmin, np.min(data_set))
        vmax = max(vmax, np.max(data_set))
    return vmin, vmax


def plot_images(data_sets, xs, ps, axes, xlabels, ylabels, titles):
    vmin, vmax = get_plotting_limits(data_sets)
    for i, data_set in enumerate(data_sets):
        ax = axes[i]
        ax.pcolormesh(xs, ps, data_set, vmin=vmin, vmax=vmax)
        ax.set_xlabel(xlabels[i])
        ax.set_ylabel(ylabels[i])
        ax.set_title(titles[i])


def plot_coherent_state_tomography_Q_functions(data_Q_function,
                                               reconstructed_state,
                                               xs,
                                               ps,
                                               noise_data=None,
                                               noise_photon_number=None,
                                               ideal_state=None,
                                               fidelities=None):
    reconstructed_qfunc = qutip.qfunc(reconstructed_state, xs, ps, g=2)
    ideal_qfunc = qutip.qfunc(ideal_state, xs, ps, g=2)

    # If the noise photons were provided, reconstruct the original data from
    # the reconstructed ideal state and a convolution (assuming a thermal noise
    # state). Also grab the noise state's Q function.
    reconstructed_noisy_Q_function = None
    noise_state_Q_function = None
    if noise_photon_number is not None:
        mesh_size = xs[1] - xs[0]
        convolution_sigma = (noise_photon_number / 2)**(1/2) / (mesh_size)
        reconstructed_noisy_Q_function = \
            ndimage.gaussian_filter(ideal_qfunc,
                                    convolution_sigma,
                                    mode='nearest')

        reconstructed_noisy_Q_function = \
            evaluate_thermally_noisy_Q_function(reconstructed_state,
                                                noise_photon_number,
                                                xs,
                                                ps)

        dim = reconstructed_state.shape[0]
        noise_state = qutip.thermal_dm(dim, noise_photon_number)
        noise_state_Q_function = qutip.qfunc(noise_state, xs, ps, g=2)

    # Plot the data and the reconstructed data if the noise state was supplied.
    num_data_sets = 1
    data_sets = [data_Q_function]
    xlabels = ['X']
    ylabels = ['P']
    titles = ['Original Input Data Q Function']
    if reconstructed_noisy_Q_function is not None:
        data_sets.append(reconstructed_noisy_Q_function) 
        xlabels.append('X')
        ylabels.append('Y')
        titles.append('Reconstructed Noisy Input Data Q Function')
        num_data_sets = 2
    fig, ax = plt.subplots(1, num_data_sets, figsize=(5*num_data_sets, 5))
    if num_data_sets == 1:
        ax = [ax]
    plot_images(data_sets, xs, ps, ax, xlabels, ylabels, titles)

    # Plot the reconstructed ideal Q function and ideal Q function if its
    # provided.
    num_data_sets = 1
    data_sets = [reconstructed_qfunc]
    xlabels = ['X']
    ylabels = ['P']
    titles = ['Reconstructed Ideal Input Data Q Function']
    if ideal_state is not None:
        data_sets.append(ideal_qfunc) 
        xlabels.append('X')
        ylabels.append('Y')
        titles.append('Ideal Input Data Q Function')
        num_data_sets = 2
    fig, ax = plt.subplots(1, num_data_sets, figsize=(5*num_data_sets, 5))
    if num_data_sets == 1:
        ax = [ax]
    plot_images(data_sets, xs, ps, ax, xlabels, ylabels, titles)

    # If the noise data exists plot it next to the ideal noise Q function
    if noise_data is not None:
        num_data_sets = 1
        data_sets = [noise_data]
        xlabels = ['X']
        ylabels = ['P']
        titles = ['Original Noise Q Function']
        if noise_state_Q_function is not None:
            data_sets.append(noise_state_Q_function) 
            xlabels.append('X')
            ylabels.append('Y')
            titles.append('Reconstructed Noise Q Function')
            num_data_sets = 2
        fig, ax = plt.subplots(1, num_data_sets, figsize=(5*num_data_sets, 5))
        if num_data_sets == 1:
            ax = [ax]
        plot_images(data_sets, xs, ps, ax, xlabels, ylabels, titles)

    # Plot the fidelities per iteration if it's provided
    if fidelities is not None:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.plot(range(len(fidelities)), fidelities)
        ax.set_xlabel('Iterations')
        ax.set_ylabel('Fidelity')
        ax.set_title('Fidelity of reconstructed state per iteration')
    
