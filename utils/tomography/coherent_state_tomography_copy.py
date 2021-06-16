import numpy as np
import qutip as qutip
import time as time

from matplotlib import pyplot as plt
from scipy import ndimage
from scipy import special

from multiprocessing import Pool
from itertools import product
import sys
import os
sys.path.append(os.path.abspath('../'))
from math_utils import linear_algebra_utils as linalg_utils
from visualization import state_visualization as state_vis

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
    

def generate_coherent_state_POVM(max_x, mesh_size, dimension, noise_base=None, cutoff=0):
    """Generates a square mesh of coherent state POVM elements centered at the
    origin. Because coherent states are overcomplete, any state can be written
    diagonally in the coherent state representation.

    If a noise state is supplied via noise_base, the POVM is modified so that
    each coherent state in the noiseless POVM inherits the fluctuation
    statistics of the noisy state (read: the POVM mesh is instead the noise
    state displaced to every point of the specified lattice in phase space.)"""
    s = time.time()
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
            # noisy_POVM_element = noisy_POVM_element.tidyup(atol=cutoff)
            noisy_POVM_element = 1/2 * noisy_POVM_element \
                                    + 1/2 * noisy_POVM_element.dag()
            # noisy_POVM_element = noisy_POVM_element.trunc_neg()
            noisy_POVM_row.append(noisy_POVM_element)
        noisy_POVM.append(noisy_POVM_row)
    
    e = time.time()
    print('Generating coherent state mesh took ' + str(e - s) + ' s.')
    return noisy_POVM, alpha_mesh, xs, ps


def truncate_povm(povm, dim):
    truncated_povm = []
    for i, povm_row in enumerate(povm):
        truncated_povm_row = []
        for j, povm in enumerate(povm_row):
            truncated_povm_element = qutip.Qobj(np.array(povm)[:dim,:dim])
            truncated_povm_row.append(truncated_povm_element)
        truncated_povm.append(truncated_povm_row)
    return truncated_povm


def sum_povm_elements(povm):
    """Sums over povm elements and returns sum"""
    dim = povm[0][0].shape[0]
    povm_sum = qutip.Qobj(np.zeros((dim, dim)))

    for row in povm:
        for povm_element in row:
            povm_sum += povm_element

    return 1/2 * povm_sum + 1/2 * povm_sum.dag()


def extract_G_inv_from_POVM(povm):
    """Returns the inverse of the sum of the povm elements."""
    povm_sum = sum_povm_elements(povm)
    G_inv = qutip.Qobj(linalg_utils.svd_inverse(povm_sum))

    #G_inv = 1/2 * G_inv + 1/2 * G_inv.dag()

    return G_inv


def extract_G_inv_from_flattened_POVM(povm):
    povm_sum = sum(povm)
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
                   frequency_threshold=0,
                   data_threshold=0,
                   cutoff=0):
    """Evaluates the iterative state update operator."""
    R = qutip.Qobj(np.zeros(state.shape))
    for i, row in enumerate(POVM_mesh):
        for j, POVM_ij in enumerate(row):
            p_ij = qutip.expect(POVM_ij, state)
            f_ij = measured_POVM_frequencies[i][j]
            if np.abs(p_ij) > frequency_threshold:
                R += POVM_ij * f_ij / p_ij

    # R = R.trunc_neg()
    R = R.tidyup(atol=cutoff)
    R = 1/2 * R + 1/2 * R.dag()

    return R


def MLE_evaluate_R_flattened(state,
                             POVM_mesh, 
                             measured_POVM_frequencies,
                             frequency_threshold=0,
                             data_threshold=0,
                             cutoff=0):
    """Evaluates the iterative state update operator."""
    R = qutip.Qobj(np.zeros(state.shape))
    for i, povm_element in enumerate(POVM_mesh):
        f_i = measured_POVM_frequencies[i]
        p_i = qutip.expect(povm_element, state)
        if np.abs(p_ij) > frequency_threshold:
            R += POVM_ij * f_i / p_i

    # R = R.trunc_neg()
    R = R.tidyup(atol=cutoff)
    R = 1/2 * R + 1/2 * R.dag()

    return R


def compute_R_component(state, povm_element, f_ij):
    p_ij = qutip.expect(povm_element, state)
    if np.abs(p_ij) > 0:
        return povm_element * f_ij / p_ij
    else:
        return 0 * povm_element


def MLE_evaluate_R_parallel(state,
                            POVM_mesh, 
                            measured_POVM_frequencies,
                            frequency_threshold=0,
                            data_threshold=0,
                            cutoff=0,
                            num_processes=1):
    """Evaluates the iterative state update operator."""
    R = qutip.Qobj(np.zeros(state.shape))
    povm_flattened = [povm_element for povm_p_row in POVM_mesh \
                        for povm_element in povm_p_row]
    f_ij_flattened = [f_ij for f_i in measured_POVM_frequencies for f_ij in f_i]
    state_stretched = [state] * len(f_ij_flattened)

    pool = Pool(processes=num_processes)
    result = pool.starmap(compute_R_component,
                          zip(state_stretched, povm_flattened, f_ij_flattened))
    pool.close()
    R = sum(result)

    R = R.tidyup(atol=cutoff)
    R = 1/2 * R + 1/2 * R.dag()

    return R


def generate_joint_povm(POVM_mesh, number_of_modes):
    povms_flattened = [povm for povm_row in POVM_mesh for povm in povm_row]
    povm_sets = product(povms_flattened, repeat=number_of_modes)
    joint_povm = []
    for s in povm_sets:
        joint_povm.append(qutip.tensor(*s))
    return joint_povm


def thin_povm(povm, freqs, cutoff=0.0):
    print('Number of povm elements prior to thinning: ' + str(len(povm)))
    indices = []
    for i, povm_element in enumerate(povm):
        if np.less(np.array(povm_element), cutoff).all():
            indices.append(i)
    thinned_povm = []
    thinned_freqs = []
    for i, _ in enumerate(povm):
        if i not in indices:
            thinned_povm.append(povm[i])
            thinned_freqs.append(freqs[i])
    
    print('Number of povm elements after thinning: ' + str(len(thinned_povm)))
    return thinned_povm, thinned_freqs


def MLE_evaluate_R_two_photon_temp(state,
                                   POVM_mesh,
                                   measured_POVM_frequencies,
                                   frequency_threshold=0,
                                   data_threshold=0,
                                   cutoff=0):
    dim = state.dims[0][0]
    zero_state = qutip.Qobj(np.zeros((dim, dim)))
    R = qutip.tensor(zero_state, zero_state)

    for i, joint_POVM in enumerate(POVM_mesh):
        p_ijkl = qutip.expect(joint_POVM, state)
        f_ijkl = measured_POVM_frequencies[i]
        if np.abs(p_ijkl) > frequency_threshold:
            R += joint_POVM * f_ijkl / p_ijkl
    
    R = R.tidyup(atol=cutoff)
    R = 1 /2 * R + 1/2 * R.dag()
    return R


def MLE_evaluate_R_two_photon(state,
                              POVM_mesh, 
                              measured_POVM_frequencies,
                              frequency_threshold=0,
                              data_threshold=0,
                              cutoff=0):
    """Evaluates the iterative state update operator."""
    dim = state.dims[0][0]
    zero_state = qutip.Qobj(np.zeros((dim, dim)))
    R = qutip.tensor(zero_state, zero_state)
    for i, rowA in enumerate(POVM_mesh):
        for j, POVM_ij_A in enumerate(rowA):
            for k, rowB in enumerate(POVM_mesh):
                for l, POVM_kl_B in enumerate(rowB):
                    joint_POVM = qutip.tensor(POVM_ij_A, POVM_kl_B)
                    p_ijkl = qutip.expect(joint_POVM, state)
                    f_ijkl = measured_POVM_frequencies[i][j][k][l]
                    if np.abs(p_ijkl) > frequency_threshold:
                        R += joint_POVM * f_ijkl / p_ijkl

    # R = R.trunc_neg()
    R = R.tidyup(atol=cutoff)
    R = 1/2 * R + 1/2 * R.dag()

    return R


def compute_R_component_two_photon(state, povm_element_pair, f_ijkl):
    POVM_ij_A = povm_element_pair[0]
    POVM_kl_B = povm_element_pair[1]
    joint_povm_element = qutip.tensor(POVM_ij_A, POVM_kl_B)
    p_ijkl = qutip.expect(joint_povm_element, state)
    if np.abs(p_ijkl) > 0:
        return joint_povm_element * f_ijkl / p_ijkl
    else:
        return 0 * joint_povm_element


def MLE_evaluate_R_two_photon_parallel(state,
                                       POVM_mesh, 
                                       measured_POVM_frequencies,
                                       frequency_threshold=0,
                                       data_threshold=0,
                                       cutoff=0,
                                       num_processes=1):
    """Evaluates the iterative state update operator."""
    dim = state.dims[0][0]

    povms_flattened = [povm for povm_row in POVM_mesh for povm in povm_row]
    povm_pairs = product(povms_flattened, repeat=2)

    f_ijkl_flattened = [f_ijkl for chunk in measured_POVM_frequencies for \
                            block in chunk for row in block for f_ijkl in row]
    state_stretched = [state] * len(f_ijkl_flattened)

    s = time.time()
    pool = Pool(processes=num_processes)
    result = pool.starmap(compute_R_component_two_photon,
                          zip(state_stretched, povm_pairs, f_ijkl_flattened))
    pool.close()
    R = sum(result)
    e = time.time()
    print('Real time taken to compute R in parallel is: ' + str(e - s) + ' s')

    R = R.tidyup(atol=cutoff)
    R = 1/2 * R + 1/2 * R.dag()

    return R


def MLE_update_state_estimate(current_state, 
                              R, 
                              G_inv,
                              identity_mixin=0,
                              cutoff=0):
    """Updates the MLE best estimate state."""
    R = (1 - identity_mixin) * R
    if not R.isherm:
        print('Ooooops')
    step = 1/2 * G_inv * R * current_state + 1/2 * current_state.dag() * R.dag() * G_inv.dag()
    step = (1/2 * step + 1/2 * step.dag()).unit()
    hold = identity_mixin * current_state
    updated_state = hold + step * (1 - identity_mixin)
    
    updated_state = updated_state.tidyup(atol=cutoff)
    # updated_state = updated_state.trunc_neg()
    updated_state = 1/2 * updated_state + 1/2 * updated_state.dag()

    return updated_state.unit()


def perform_coherent_state_MLE(povm,
                               measured_POVM_frequencies, 
                               number_of_iterations, 
                               rho0=None, 
                               rho_ideal=None,
                               identity_mixin=0,
                               frequency_threshold=0,
                               data_threshold=0,
                               cutoff=0,
                               number_of_photons=1):
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
    s = time.time()
    # Extract the POVM completion operator
    G_inv = extract_G_inv_from_POVM(povm)
    if number_of_photons == 2:
        G_inv = qutip.tensor(G_inv, G_inv)
    
    # Initialize unbiased mixed state as MLE initial state
    dim = povm[0][0].shape[0] # ugly way to extract dimension
    state = 1/dim * qutip.qeye(dim)
    if number_of_photons == 2:
        state = qutip.tensor(state, state)
    
    if rho0 is not None:
        state = rho0
    
    intermediate_fidelities = []
    log_likelihoods = []
    bad_states = []
    traces = []
    R_mins = []
    R_maxes = []

    if number_of_photons == 2:
        print('Generating joint povm...')
        s = time.time()
        povm = generate_joint_povm(povm, number_of_photons)
        f_ijkl_flattened = [f_ijkl for chunk in measured_POVM_frequencies for \
                            block in chunk for row in block for f_ijkl in row]
        e = time.time()
        print('Generating joint povm mesh took ' + str(e - s) + ' s.')

    # Run MLE
    for i in range(number_of_iterations):
        if number_of_photons == 1:
            R = MLE_evaluate_R(state, 
                               povm, 
                               measured_POVM_frequencies,
                               frequency_threshold=frequency_threshold,
                               data_threshold=data_threshold,
                               cutoff=cutoff)
        elif number_of_photons == 2:
            R = MLE_evaluate_R_two_photon_temp(state, 
                                       povm, 
                                       f_ijkl_flattened,
                                       frequency_threshold=frequency_threshold,
                                       data_threshold=data_threshold,
                                       cutoff=cutoff)
        
        state = MLE_update_state_estimate(state, 
                                          R, 
                                          G_inv, 
                                          identity_mixin=identity_mixin,
                                          cutoff=cutoff)
        if rho_ideal is not None:
            intermediate_fidelities.append(qutip.fidelity(state, rho_ideal))
        traces.append(state.tr())

        if number_of_photons == 1:
            log_likelihood = evaluate_log_likelihood(measured_POVM_frequencies,
                                                     state,
                                                     povm)
        # elif number_of_photons == 2:
#            log_likelihood = evaluate_log_likelihood_two_photon(
#                                 measured_POVM_frequencies,
#                                 state,
#                                 povm)

        log_likelihoods.append(log_likelihood)
    e = time.time()
    print('Non-inlined, unflattened, untruncated coherent state MLE of ' + \
              str(number_of_photons) + ' photon mode(s) and a Fock space ' + \
              ' dimension of ' + str(dim) + ' took ' + str(e - s) + ' seconds ' + \
              ' for ' + str(number_of_iterations) + '.')
    
    if rho_ideal is not None:
        return state, intermediate_fidelities, log_likelihoods# , bad_states, traces, R_mins, R_maxes, log_likelihoods
    
    return state, log_likelihoods


def perform_coherent_state_MLE_inline(povm,
                                      measured_POVM_frequencies, 
                                      G_inv,
                                      number_of_iterations, 
                                      rho0=None, 
                                      rho_ideal=None,
                                      identity_mixin=0,
                                      frequency_threshold=0,
                                      data_threshold=0,
                                      cutoff=0,
                                      number_of_photons=1):
    """This is essentially the same as `perform_coherent_state_MLE` but it works
    with flattened POVMs of any size (no need to switch on photon number)."""
    s = time.time()
    
    # Initialize unbiased mixed state as MLE initial state
    dim = povm[0].shape[0] # ugly way to extract dimension
    subspace_identities = []
    for dimension in povm[0].dims[0]:
        subspace_identities.append(qutip.qeye(dimension))
    state = 1/dim * qutip.tensor(*subspace_identities)
        
    if rho0 is not None:
        state = rho0
    
    intermediate_fidelities = []
    log_likelihoods = []
    bad_states = []
    traces = []
    R_mins = []
    R_maxes = []

    subspace_zeros = []
    print(state.dims)
    for shape in state.dims[0]:
        subspace_zeros.append(qutip.Qobj(np.zeros((shape, shape))))
    print(len(subspace_zeros))
    zeros = qutip.tensor(*subspace_zeros)

    print(zeros.shape)
    print(G_inv.dims)

    # Run MLE
    for i in range(number_of_iterations):
        R = qutip.Qobj(zeros)
        for i, povm_element in enumerate(povm):
            f_i = measured_POVM_frequencies[i]
            p_i = qutip.expect(povm_element, state)
            if np.abs(p_i) > frequency_threshold:
                R += povm_element * f_i / p_i

        # R = R.trunc_neg()
        R = R.tidyup(atol=cutoff)
        R = 1/2 * R + 1/2 * R.dag()
        
        R = (1 - identity_mixin) * R
        if not R.isherm:
            print('Ooooops')
        step = 1/2 * G_inv * R * state + 1/2 * state.dag() * R.dag() * G_inv.dag()
        step = (1/2 * step + 1/2 * step.dag()).unit()
        hold = identity_mixin * state
        updated_state = hold + step * (1 - identity_mixin)
        
        updated_state = updated_state.tidyup(atol=cutoff)
        updated_state = 1/2 * updated_state + 1/2 * updated_state.dag()
        
        state = updated_state
        
        if rho_ideal is not None:
            intermediate_fidelities.append(qutip.fidelity(state, rho_ideal))

    e = time.time()
    report = 'Inlined, flattened coherent state MLE of ' + str(number_of_photons) + \
                 ' photonic modes and truncated Fock space dimension ' + str(dim) + \
                 ' took ' + str(e - s) + ' seconds.'
    print(report)
    
    if rho_ideal is not None:
        return state, intermediate_fidelities
    
    return state, log_likelihoods


def evaluate_log_likelihood(data, state, povm):
    log_likelihood = 0
    for i, row in enumerate(povm):
        for j, povm_element in enumerate(row):
            p_ij = qutip.expect(povm_element, state)
            f_ij = data[i][j]
            p_ij = np.abs(p_ij)
            if p_ij > 0:
                log_likelihood += f_ij * np.log(p_ij)
    return log_likelihood


def evaluate_log_likelihood_two_photon(data, state, povm):
    log_likelihood = 0
    for i, rowA in enumerate(povm):
        for j, povm_element_A in enumerate(rowA):
            for k, rowB in enumerate(povm):
                for l, povm_element_B in enumerate(rowB):
                    joint_povm_element = qutip.tensor(povm_element_A,
                                                      povm_element_B)
                    p_ijkl = qutip.expect(joint_povm_element, state)
                    f_ijkl = data[i][j][k][l]
                    p_ijkl = np.abs(p_ijkl)
                    if p_ijkl > 0:
                        log_likelihood += f_ijkl * np.log(p_ijkl)
    return log_likelihood


def inverse_bernoulli_transform(rho, eta, max_lost_photons):
    """Perform the inverse Bernoulli transform up to a redistribution of
    `max_lost_photons` assuming a loss rate `eta` on state `rho`."""
    dim = rho.shape[0]
    B_numbers = np.zeros((dim + max_lost_photons, dim + max_lost_photons))
    for i, _ in enumerate(B_numbers):
        for j, _ in enumerate(B_numbers[0]):
            B_numbers[i][j] = special.comb(i, j)
    
    inverted_rho = np.zeros(rho.shape, dtype=np.complex128)
    for k in range(max_lost_photons + 1):
        for m, _ in enumerate(inverted_rho):
            for n, _ in enumerate(inverted_rho[0]):
                if n + k < dim and m + k < dim:
                    eta_coeff = (eta)**(-(n + m)/2) * (1 - 1/eta)**k
                    B_coeff = B_numbers[m + k][m]**(1/2) * B_numbers[n + k][n]**(1/2)
                    inverted_rho[m][n] += rho[n + k][0][m + k] * eta_coeff * B_coeff
    
    return inverted_rho


def bernoulli_transform(rho, eta, max_lost_photons):
    """Perform the forward Bernoulli transform on rho. Defined identically to
    the inverse bernoulli transform but with eta -> eta^-1."""
    return inverse_bernoulli_transform(rho, 1/eta, max_lost_photons)


def povm_bernoulli_transform(povm, eta, max_lost_photons):
    """Transform the povm elements of a povm to compensate for known photon
    detection efficiency `eta`"""
    transformed_povm = []
    for row in povm:
        transformed_povm_row = []
        for povm_element in row:
            transformed_povm_row.append(
                povm_element_bernoulli_transform(povm_element,
                                                 eta,
                                                 max_lost_photons))
        transformed_povm.append(transformed_povm_row)
    return transformed_povm


def povm_element_bernoulli_transform(povm_element, eta, max_lost_photons):
    """Transform a single povm element into its corresponding compensatory povm
    element according with a photon detection efficiency of `eta`."""
    dim = povm_element.shape[0]
    transformed_povm_element = np.zeros(povm_element.shape, dtype=np.complex128)

    # Generate necessary combinatoric values
    s = time.time()
    B_numbers = np.zeros((dim + max_lost_photons, dim + max_lost_photons))
    for i, _ in enumerate(B_numbers):
        for j, _ in enumerate(B_numbers[0]):
            B_numbers[i][j] = special.comb(i, j)
    e = time.time()
    print('Time generating Bernoulli numbers: ' + str(e - s) + ' s')
    
    # Transform the POVM element
    for k in range(max_lost_photons + 1):
        for m in range(dim):
            for n in range(dim):
                if n + k < dim and m + k < dim:
                    p_mn = povm_element[m][0][n]
                    eta_coeff = eta**((m + n)/2) * (1 - eta)**k
                    B_coeff = B_numbers[n + k][k]**(1/2) * B_numbers[m + k][k]**(1/2)
                    transformed_povm_element[n + k][m + k] += \
                        eta_coeff * B_coeff * p_mn

    return qutip.Qobj(transformed_povm_element)


def generate_bernoulli_coefficient_tensor(eta, dim, max_lost_photons):
    B_kmn = []
    for k in range(max_lost_photons + 1):
        B_mn = np.zeros((dim, dim))
        for m in range(dim):
            for n in range(dim):
                if m + k < dim and n + k < dim:
                    eta_coeff = eta**((m + n)/2) * (1 - eta)**k
                    B_coeff = special.comb(m + k, k) * special.comb(n + k, k)
                    B_coeff = B_coeff**(1/2)
                    B_mn[m + k][n + k] = eta_coeff * B_coeff
        B_kmn.append(B_mn)
    return np.array(B_kmn)


def stack_shifted_povm_elements(povm_element, max_lost_photons):
    p_kmn = []
    shape = povm_element.shape
    povm_element = np.array(povm_element)

    # Treat the k=0 case differently because of indexing issue with -0
    p_kmn.append(povm_element)
    for k in range(1, max_lost_photons + 1):
        p_mn = np.zeros(shape, dtype=np.complex128)
        p_mn[k:,k:] = povm_element[:-k,:-k]
        p_kmn.append(p_mn)
    return np.array(p_kmn)


def fast_povm_element_bernoulli_transform(povm_element, B_kmn, max_lost_photons):
    p_kmn = stack_shifted_povm_elements(povm_element, max_lost_photons)
    p_B = p_kmn * B_kmn
    transformed_povm_element = np.sum(p_B, axis=0)
    return qutip.Qobj(transformed_povm_element)


def fast_povm_bernoulli_transform(povm, eta, max_lost_photons):
    dim = povm[0][0].shape[0]
    transformed_povm = []
    B_kmn = generate_bernoulli_coefficient_tensor(eta, dim, max_lost_photons)
    for row in povm:
        transformed_povm_row = []
        for povm_element in row:
            transformed_povm_row.append(
                fast_povm_element_bernoulli_transform(povm_element,
                                                      B_kmn,
                                                      max_lost_photons))
        transformed_povm.append(transformed_povm_row)
    return transformed_povm


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
    if ideal_state is not None:
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

    # Plot the Fock basis amplitudes
    state_vis.plot_fock_basis_amplitudes(reconstructed_state)

    
### SAVING UTILITIES ####
def save_povm(path, povm, alphas, xs, ps, noise=None):
    """Fine / wide povms take over an hour to generate, thus it's useful to be
    able to save them and reload them quickly."""
    povm_arrays = []
    for row in povm:
        for povm in row:
            povm_arrays.append(np.array(povm))
    povm_arrays = np.array(povm_arrays)

    if noise is None:
	    np.savez(path,
	             povm_arrays,
	             alphas,
	             xs,
	             ps)
    else:
        noise = np.array(noise)
        np.savez(path,
                 povm_arrays,
                 alphas,
                 xs,
                 ps,
                 noise)


def load_povm(path):
    npz = np.load(path)
    povm_arrays = npz['arr_0']
    alphas = npz['arr_1']
    xs = npz['arr_2']
    ps = npz['arr_3']
    
    mesh_shape = alphas.shape

    povms = []
    for i in range(mesh_shape[0]):
        povm_row = []
        for j in range(mesh_shape[1]):
            povm = povm_arrays[i*mesh_shape[0] + j]
            povm_row.append(qutip.Qobj(povm))
        povms.append(povm_row)
    
    if 'arr_4' in npz.files:
        noise_state = npz['arr_4']
        noise_state = qutip.Qobj(noise_state)
    else:
        noise_state = None

    return povms, alphas, xs, ps, noise_state


def save_reconstruction_iteration(path,
                                  state,
                                  histogram,
                                  log_likelihoods,
                                  alphas,
                                  xs,
                                  ps,
                                  iter_range,
                                  rho0,
                                  povm_path):
    state = np.array(state)
    rho0 = np.array(rho0)
    povm_path = np.array([povm_path])
    iter_range = np.array(iter_range)
    histogram = np.array(histogram)
    np.savez(path,
             state,
             histogram,
             log_likelihoods,
             alphas,
             xs,
             ps,
             iter_range,
             rho0,
             povm_path)


def load_reconstruction_iteration(path):
    npz = np.load(path)
    state = qutip.Qobj(npz['arr_0'])
    histogram = npz['arr_1']
    log_likelihoods = npz['arr_2']
    alphas = npz['arr_3']
    xs = npz['arr_4']
    ps = npz['arr_5']
    iter_range = npz['arr_6']
    rho0 = qutip.Qobj(npz['arr_7'])
    try:
        povm_path = npz['arr_8'][0]
        povms, _, _, _, _ = load_povm(povm_path)
    except Exception as e:
        print(e)
        povms = None
    return state, histogram, log_likelihoods, alphas, xs, ps, iter_range, rho0, povms

