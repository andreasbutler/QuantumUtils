import os
import numpy as np
import scipy.optimize as opt
from qutip import *
from matplotlib import pyplot as plt

from qutip_utils import qutip_utils
from qutip_utils import pulse_utils
from qutip_utils import dsp_utils

# From Labber's random clifford generator
def add_singleQ_clifford(index, gate_seq):
    """Add single qubit clifford (24)."""
    length_before = len(gate_seq)
    # Paulis
    if index == 0:
        gate_seq.append('id')
    elif index == 1:
        gate_seq.append('xp')
    elif index == 2:
        gate_seq.append('yp')
    elif index == 3:
        gate_seq.append('yp')
        gate_seq.append('xp')

    # 2pi/3 rotations
    elif index == 4:
        gate_seq.append('x2p')
        gate_seq.append('y2p')
    elif index == 5:
        gate_seq.append('x2p')
        gate_seq.append('y2m')
    elif index == 6:
        gate_seq.append('x2m')
        gate_seq.append('y2p')
    elif index == 7:
        gate_seq.append('x2m')
        gate_seq.append('y2m')
    elif index == 8:
        gate_seq.append('y2p')
        gate_seq.append('x2p')
    elif index == 9:
        gate_seq.append('y2p')
        gate_seq.append('x2m')
    elif index == 10:
        gate_seq.append('y2m')
        gate_seq.append('x2p')
    elif index == 11:
        gate_seq.append('y2m')
        gate_seq.append('x2m')

    # pi/2 rotations
    elif index == 12:
        gate_seq.append('x2p')
    elif index == 13:
        gate_seq.append('x2m')
    elif index == 14:
        gate_seq.append('y2p')
    elif index == 15:
        gate_seq.append('y2m')
    elif index == 16:
        gate_seq.append('x2m')
        gate_seq.append('y2p')
        gate_seq.append('x2p')
    elif index == 17:
        gate_seq.append('x2m')
        gate_seq.append('y2m')
        gate_seq.append('x2p')

    # Hadamard-Like
    elif index == 18:
        gate_seq.append('xp')
        gate_seq.append('y2p')
    elif index == 19:
        gate_seq.append('xp')
        gate_seq.append('y2m')
    elif index == 20:
        gate_seq.append('yp')
        gate_seq.append('x2p')
    elif index == 21:
        gate_seq.append('yp')
        gate_seq.append('x2m')
    elif index == 22:
        gate_seq.append('x2p')
        gate_seq.append('y2p')
        gate_seq.append('x2p')
    elif index == 23:
        gate_seq.append('x2m')
        gate_seq.append('y2p')
        gate_seq.append('x2m')

    # For interleaved gates use index + 24
    elif index == 25:
        gate_seq.append('xp_interleaved')

    elif index == 28:
        gate_seq.append('x2p_interleaved')

    elif index == -1:
        gate_seq.append('noise')

    else:
        raise Exception('Index not in gate set')


def add_pulses_by_key(key, pulses, x, y, z):
    if key is 'id':
        x = np.concatenate((x, pulses['identity']))
        y = np.concatenate((y, pulses['identity']))
        z = np.concatenate((z, pulses['identity']))
    elif key is 'xp':
        if 'x_pi' in pulses:
            x = np.concatenate((x, pulses['x_pi']))
            y = np.concatenate((y, pulses['x_pi_derivative']))
            z = np.concatenate((z, pulses['x_pi_detuning']))
        else:
            x = np.concatenate((x, pulses['pi']))
            y = np.concatenate((y, pulses['pi_derivative']))
            z = np.concatenate((z, pulses['pi_detuning']))
    elif key is 'yp':
        x = np.concatenate((x, -pulses['pi_derivative']))
        y = np.concatenate((y, pulses['pi']))
        z = np.concatenate((z, pulses['pi_detuning']))
    elif key is 'x2p':
        x = np.concatenate((x, pulses['pi_half']))
        y = np.concatenate((y, pulses['pi_half_derivative']))
        z = np.concatenate((z, pulses['pi_half_detuning']))
    elif key is 'x2m':
        x = np.concatenate((x, -pulses['pi_half']))
        y = np.concatenate((y, -pulses['pi_half_derivative']))
        z = np.concatenate((z, pulses['pi_half_detuning']))
    elif key is 'y2p':
        x = np.concatenate((x, -pulses['pi_half_derivative']))
        y = np.concatenate((y, pulses['pi_half']))
        z = np.concatenate((z, pulses['pi_half_detuning']))
    elif key is 'y2m':
        x = np.concatenate((x, pulses['pi_half_derivative']))
        y = np.concatenate((y, -pulses['pi_half']))
        z = np.concatenate((z, pulses['pi_half_detuning']))
    elif key is 'noise':
        x = np.concatenate((x, pulses['x_noise']))
        y = np.concatenate((y, pulses['y_noise']))
        z = np.concatenate((z, pulses['z_noise']))
    return x, y, z


def build_full_pulse_sequence(gate_seq, pulse_len, pulse_dictionary, sample_rate):
    x = np.zeros((0))
    y = np.zeros((0))
    z = np.zeros((0))
    for key in gate_seq:
        x, y, z = add_pulses_by_key(key, pulse_dictionary, x, y, z)
    
    # calculate total time
    num_gates = len(gate_seq)
    total_time = pulse_len*num_gates
    # for now, cumbersomely assume that the noise gates are as long as the actual gates
    times = np.linspace(0, total_time, sample_rate*total_time)
    return x, y, z, times


def build_full_pulse_sequence_with_dict(gate_seq,
                                        clifford_dictionary,
                                        sample_rate):
    """Given a dictionary of clifford labels to x, y, z pulse definitions,
    converts a sequence of clifford labels into the concatenation of their
    pulse representation."""
    x = np.zeros(0)
    y = np.zeros(0)
    z = np.zeros(0)
    for label in gate_seq:
        x = np.concatenate((x, clifford_dictionary[label][0]))
        y = np.concatenate((y, clifford_dictionary[label][1]))
        z = np.concatenate((z, clifford_dictionary[label][2]))

    times = dsp_utils.get_times(x, sample_rate)
    return x, y, z, times


def find_and_insert_clifford_inverse(gate_seq):
    dict_m1QBGate = {
        'id': np.matrix('1,0;0,1'),
        'x2p': 1/np.sqrt(2)*np.matrix('1,-1j;-1j,1'),
        'x2m': 1/np.sqrt(2)*np.matrix('1,1j;1j,1'),
        'y2p': 1/np.sqrt(2)*np.matrix('1,-1;1,1'),
        'y2m': 1/np.sqrt(2)*np.matrix('1,1;-1,1'),
        'z2p': np.matrix('1,0;0,1j'),
        'z2m': np.matrix('1,0;0,-1j'),
        'xp': np.matrix('0,-1j;-1j,0'),
        'xm': np.matrix('0,1j;1j,0'),
        'yp': np.matrix('0,-1;1,0'),
        'ym': np.matrix('0,1;-1,0'),
        'zp': np.matrix('1,0;0,-1'),
        'zm': np.matrix('-1,0;0,1')
    }
    for i in range(0,24):
        temp = gate_seq.copy()
        add_singleQ_clifford(i, temp)
        scratch = dict_m1QBGate['id']
        for key in temp:
            if 'interleaved' in key:
                key = key[:-12]
            scratch = np.matmul(scratch, dict_m1QBGate[key])
        if (np.isclose(scratch, dict_m1QBGate['id'])).all():
            add_singleQ_clifford(i, gate_seq)
            return i
        if (np.isclose(scratch, -dict_m1QBGate['id'])).all():
            add_singleQ_clifford(i, gate_seq)
            return i
    return 0


def generate_random_clifford_sequence(length, noisy=False, interleaved_gate=None):
    random_indices = np.random.randint(0, 24, length)
    if interleaved_gate is not None:
        temp = []
        for index in random_indices:
            temp.append(index)
            temp.append(interleaved_gate)
        random_indices = temp
    ideal_gate_seq = []
    for index in random_indices:
        add_singleQ_clifford(index, ideal_gate_seq)
    inverse_index = find_and_insert_clifford_inverse(ideal_gate_seq)
    
    if noisy is not True:
        return ideal_gate_seq
    
    # If we are artificially inserting our own noise, it was necessary to first
    # compute the ideal clifford sequence so we could find the ideal final
    # inverse gate.
    noisy_indices = []
    noisy_gate_sequence = []
    for i in range(len(random_indices)):
        noisy_indices.append(random_indices[i])
        noisy_indices.append(-1) # noise flag
    # Insert the final inverting gate and its associated noise
    noisy_indices.append(inverse_index)
    noisy_indices.append(-1)
    # Turn the noisy indices into pulse labels
    for index in noisy_indices:
        add_singleQ_clifford(index, noisy_gate_sequence)
    return noisy_gate_sequence


def plot_pulse_train_results(states, projectors, pulse_len, times, x, y, z):
    DRAG_xs, DRAG_ys, DRAG_zs, _, _ = \
        qutip_utils.extract_qubit_info(states, 0, 1)
    qutip_utils.plot_bloch_profiles(DRAG_xs, DRAG_ys, DRAG_zs)
    
    populations = qutip_utils.extract_state_populations(states, projectors)
    
    fig, axs = plt.subplots(2, 1)
    # Plot pulses
    axs[0].plot(times, x, color='red')
    axs[0].plot(times, y, color='purple')
    axs[0].plot(times, z, color='gold')
    mezcla = np.concatenate((np.concatenate((x,y)),z))
    minimum = np.amin(mezcla) - 0.01
    maximum = np.amax(mezcla) + 0.01
    axs[0].set_ylim((minimum, maximum))
    axs[0].set_ylabel('Pulse strength (au)')
    axs[0].legend(['X', 'Y', 'Z'])
    for i in range(int((times[-1]+1)/pulse_len) + 1):
        axs[0].plot([i*pulse_len, i*pulse_len], [minimum, maximum], c='green', linestyle='dashed', linewidth=0.5)
    
    
    for pop in populations:
        axs[1].plot(times, pop)
    for i in range(int((times[-1]+1)/pulse_len) + 1):
        axs[1].plot([i*pulse_len, i*pulse_len], [0, 1], c='green', linestyle='dashed', linewidth=0.5)
    axs[1].set_xlabel('Time (ns)')
    axs[1].set_ylabel('Population')
    axs[1].legend(['g', 'e', 'f', 'h'])
    fig.tight_layout()


def run_rb_iteration(length, pulse_info, simulation_parts, noisy=False, interleaved_gate=None):
    gate_sequence = generate_random_clifford_sequence(length, noisy=noisy, interleaved_gate=interleaved_gate)
    pulse_len = pulse_info['pulse_len']
    pulse_dictionary = pulse_info['pulse_dictionary']
    sampling_rate = pulse_info['sampling_rate']
    x, y, z, times = build_full_pulse_sequence(gate_sequence, pulse_len, pulse_dictionary, sampling_rate)
    H = [simulation_parts['H0'],
         [simulation_parts['Hx'], x],
         [simulation_parts['Hy'], y],
         [simulation_parts['Hz'], z]]
    psi0 = simulation_parts['psi0']
    output = mesolve(H, psi0, times)
    return (np.abs((output.states[-1])*(psi0.dag())))**2


def run_rb(lengths, iterations_per_length, pulse_info, simulation_parts, noisy=False, interleaved_gate=None):
    survival_probabilities = []
    for length in lengths:
        survival_probability = 0
        for i in range(iterations_per_length):
            survival_probability += run_rb_iteration(length,
                                                     pulse_info,
                                                     simulation_parts,
                                                     noisy=noisy,
                                                     interleaved_gate=interleaved_gate)
        survival_probabilities.append(survival_probability/iterations_per_length)
    return survival_probabilities


def run_rb_iteration_decoherence(length, pulse_info, simulation_parts, interleaved_gate=None):
    gate_sequence = generate_random_clifford_sequence(length, interleaved_gate=interleaved_gate)
    pulse_len = pulse_info['pulse_len']
    pulse_dictionary = pulse_info['pulse_dictionary']
    sampling_rate = pulse_info['sampling_rate']
    x, y, z, times = build_full_pulse_sequence(gate_sequence, pulse_len, pulse_dictionary, sampling_rate)
    H = [simulation_parts['H0'],
         [simulation_parts['Hx'], x],
         [simulation_parts['Hy'], y],
         [simulation_parts['Hz'], z]]
    psi0 = simulation_parts['psi0']
    # c_ops = simulation_parts['c_ops']
    ground_state_projector = simulation_parts['gp']
    if 'c_ops' in simulation_parts:
        c_ops = simulation_parts['c_ops']
        output = mesolve(H, psi0, times, c_ops=c_ops)
        return expect(output.states[-1], ground_state_projector)
    else:
        output = mesolve(H, psi0, times)
        return (np.abs(output.states[-1][0][0]))**2
    return expect(output.states[-1], ground_state_projector)


def run_rb_decoherence(lengths, iterations_per_length, pulse_info, simulation_parts, interleaved_gate=None):
    survival_probabilities = []
    for length in lengths:
        survival_probability = 0
        for i in range(iterations_per_length):
            survival_probability += run_rb_iteration_decoherence(length,
                                                     pulse_info,
                                                     simulation_parts,
                                                     interleaved_gate=interleaved_gate)
        survival_probabilities.append(survival_probability/iterations_per_length)
    return survival_probabilities


def build_full_pulse_sequence_interleaved_noise(gate_seq, pulse_len, pulse_dictionary, noise_pulses, sample_rate):
    x = np.zeros((0))
    y = np.zeros((0))
    z = np.zeros((0))
    for key in gate_seq:
        x, y, z = add_pulses_by_key(key, pulse_dictionary, x, y, z)
        x, y, z = add_noise(x, y, z, noise_pulses)
    
    # calculate total time
    num_gates = len(gate_seq)
    total_time = pulse_len*num_gates
    times = np.linspace(0, total_time, sample_rate*total_time)
    return x, y, z, times


def add_noise(x, y, z, noise_pulses):
    x = np.concatenate((x, noise_pulses['x']))
    y = np.concatenate((y, noise_pulses['y']))
    z = np.concatenate((y, noise_pulses['y']))


def build_generator_signals(pulse_len, sample_rate, DRAG_params):
    """ Creates the signals for a pi and pi-half pulse.

    Args:
      pulse_len: the length of the pulses in ns
      sample_rate: the sampling rate of the signal in GHz
      DRAG_params: the DRAG coefficients to use
    Returns:
      A dictionary of 1D numpy arrays of length set by pulse_len*sample_rate
      for the gaussian envelope, derivative envelope, and detuning envelope of
      pi and pi-half pulses, as well as an identity pulse of pulse_len."""
    A_pi = np.pi
    A_pi_half = np.pi/2
    
    # Arguments for envelopes
    pi_envelope_args = {
        'A': A_pi,
        'x_coeff': DRAG_params['x_coeff'],
        'y_coeff': DRAG_params['y_coeff'],
        'det_coeff': DRAG_params['det_coeff'],
        'tg': pulse_len/2,
        'tn': pulse_len/2,
        'tsigma': pulse_len/4
    }
    
    pi_half_envelope_args = {
        'A': A_pi_half,
        'x_coeff': DRAG_params['x_coeff'],
        'y_coeff': DRAG_params['y_coeff'],
        'det_coeff': DRAG_params['det_coeff'],
        'tg': pulse_len/2,
        'tn': pulse_len/2,
        'tsigma': pulse_len/4
    }
    
    times, pi_env, pi_deriv, pi_dets = \
        DRAG_utils.create_ge_envelopes(sample_rate,
                                       pulse_len,
                                       pi_envelope_args)
    times, pi_half_env, pi_half_deriv, pi_half_dets = \
        DRAG_utils.create_ge_envelopes(sample_rate, 
                                       pulse_len, 
                                       pi_half_envelope_args)
    identity = np.zeros(len(pi_env['r']))
    
    # Construct the pulse dictionary:
    pulse_dict = {
      'identity': identity,
      'pi': np.array(pi_env['r']),
      'pi_derivative': np.array(pi_deriv['r']),
      'pi_detuning': np.array(pi_dets['r']),
      'pi_half': np.array(pi_half_env['r']),
      'pi_half_derivative': np.array(pi_half_deriv['r']),
      'pi_half_detuning': np.array(pi_half_dets['r']),
    }
    return pulse_dict


def build_simple_x_noise(angle, pulse_len, sample_rate, DRAG_params):
    """ Construct the X, Y, and Z signals for an X rotation

    Though the result of this function is to make a completely general X
    rotation, it's named suggestively because it's meant to be used to
    create a simple noise channel that consists of a small rotation about the X
    axis of the Bloch sphere.
    
    Args:
      angle: the small angle of rotation about X
      pulse_len: the length of the noise channel in ns
      sample_rate: the sample rate of the signals to create in GHz
      DRAG_params: a dictionary of the first order DRAG coefficients for each
          control line. allow for DRAG on the rotation so that the infidelity of
          the noise is as engineered as possible (ie: minimize infidelity of
          the noise that comes from any source other than it being a spurious
          rotation).
    Returns:
      3 numpy 1D arrays corresponding to the noise signals."""
    envelope_args = {
        'A': angle,
        'x_coeff': DRAG_params['x_coeff'],
        'y_coeff': DRAG_params['y_coeff'],
        'det_coeff': DRAG_params['det_coeff'],
        'tg': pulse_len/2,
        'tn': pulse_len/2,
        'tsigma': pulse_len/4
    }
    _, x_noise, y_noise, z_noise = \
        DRAG_utils.create_ge_envelopes(sample_rate,
                                       pulse_len,
                                       envelope_args)
    return np.array(x_noise['r']), \
           np.array(y_noise['r']), \
           np.array(z_noise['r'])


def first_order_fitting_function(m, p, A, B):
        return A*np.power(p, m) + B


def first_order_fit(lengths, survival_probabilities):
    """ Returns the fitting parameters for first order RB.

    Args:
      lengths: A 1D array of the RB sequence lengths analyzed.
      survival_probabilities: The average survival probabilities associated
          with the lengths
    Returns:
      The parameters p, A, and B for the first order RB fit."""
    # Define initial guesses
    p = 1
    A = 1/2
    B = 1/2
    initial_param_guess = [p, A, B]
    def first_order_fitting_function(m, p, A, B):
        return A*np.power(p, m) + B
    first_order_params = opt.curve_fit(first_order_fitting_function,
                                       lengths,
                                       survival_probabilities,
                                       p0 = initial_param_guess,
                                       maxfev=10000,
                                       bounds = ([0, -1, 0], [1, 1, 1]))
    return first_order_params


def second_order_fitting_function(m, p, q, A, B, C):
        return A*np.power(p, m) + B + C*(m - 1)*(q - p*p)*np.power(p, m - 2)


def second_order_fit(lengths, survival_probabilities):
    """ Returns the fitting parameters for second order RB.

    Args:
      lengths: A 1D array of the RB sequence lengths analyzed.
      survival_probabilities: The average survival probabilities associated
          with the lengths
    Returns:
      The parameters p, q, A, B, and C for the second order RB fit."""
    # Define initial guesses
    p = 1
    q = 1/2
    A = 1/2
    B = 1/2
    C = 1/2
    initial_param_guess = [p, q, A, B, C]
    def second_order_fitting_function(m, p, q, A, B, C):
        return A*np.power(p, m) + B + C*(m - 1)*(q - p*p)*np.power(p, m - 2)
    second_order_params = opt.curve_fit(second_order_fitting_function,
                                        np.array(lengths),
                                        np.array(survival_probabilities),
                                        p0 = initial_param_guess,
                                        maxfev=10000,
                                        bounds = ([0, 0, -np.inf, -np.inf, -np.inf], \
                                                  [1, 1, np.inf, np.inf, np.inf]))
    return second_order_params


def save_simple_noise_simulation(path,
                                 filename,
                                 x_noise,
                                 y_noise,
                                 z_noise,
                                 lengths,
                                 survival_probabilities,
                                 first_order_fit_parameters,
                                 second_order_fit_parameters):
    """ A utility function for saving simulation data."""
    to_save = np.array([x_noise,
                        y_noise,
                        z_noise,
                        lengths,
                        survival_probabilities,
                        first_order_fit_parameters,
                        second_order_fit_parameters])
    full_path = os.path.join(path, filename)
    np.savetxt(full_path, to_save)


def recover_saved_simple_noise_simulation(path):
    """ A utitility function for recovering pieces of a simple simulation."""
    data = np.loadtxt(path)
    ret_dict = {
        'x_noise': data[0],
        'y_noise': data[1],
        'z_noise': data[2],
        'lengths': data[3],
        'survival_probabilities': data[4],
        'first_order_fit_parameters': data[5],
        'second_order_fit_parameters': data[6]
    }
    return ret_dict


###### SINGLE TWIRL CALCULATION UTILITIES #####################################
def generate_clifford_inverse_pair_indices():
    clifford_indices = [[i] for i in range(24)]
    clifford_labels = []
    for i in range(24):
        gate_seq = []
        add_singleQ_clifford(i, gate_seq)
        clifford_labels.append(gate_seq)
    for i in range(24):
        index = find_and_insert_clifford_inverse(clifford_labels[i])
        clifford_indices[i].append(index)
    return clifford_indices


def generate_clifford_inverse_pair_label_sets():
    clifford_indices = generate_clifford_inverse_pair_indices()
    clifford_inverse_pairs = []
    for pair in clifford_indices:
        cliff_pulse_seq = {}
        inv_pulse_seq = {}
        cliff_gate_seq = []
        inv_gate_seq = []
        add_singleQ_clifford(pair[0], cliff_gate_seq)
        add_singleQ_clifford(pair[1], inv_gate_seq)
        clifford_inverse_pairs.append((cliff_gate_seq, inv_gate_seq))
    return clifford_inverse_pairs


def perform_single_twirl(process, clifford_pulse_dictionary, dim, H):
    clifford_inverse_pair_labels = generate_clifford_inverse_pair_label_sets()
    g = qutip.basis(dim, 0)
    H0 = H[0]
    Hx = H[1]
    Hy = H[2]
    Hz = H[3]
    scratch = qutip.Qobj(np.zeros((dim, dim)))
    for clifford_labels, inverse_labels in clifford_inverse_pair_labels:
        input_state = g * g.dag()

        # Clifford transform
        for label in clifford_labels:
            x, y, z, times = build_full_pulse_sequence_with_dict([label],
                                                        clifford_pulse_dictionary,
                                                        sample_rate)
            H = [H0, [Hx, x], [Hy, y], [Hz, z]]
            output = qutip.mesolve(H, input_state, times)
            input_state = output.states[-1]
       
        # Process
        H = [H0, [Hx, process[0]], [Hy, process[1]], [Hz, process[2]]]
        times = process[3]
        output = qutip.mesolve(H, input_state, times)
        input_state = output.states[-1]

        # Inverse Clifford transform
        for label in clifford_labels:
            x, y, z, times = build_full_pulse_sequence_with_dict([label],
                                                        clifford_pulse_dictionary,
                                                        sample_rate)
            H = [H0, [Hx, x], [Hy, y], [Hz, z]]
            output = qutip.mesolve(H, input_state, times)
            input_state = output.states[-1]

        scratch += input_state
    return scratch


