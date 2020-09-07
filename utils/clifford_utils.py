import numpy as np
import qutip as qutip
from qutip_utils import pulse_utils as pulse_utils

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


def add_pulses_by_key(key, pulses, x, y, z):
    if key is 'id':
        x = np.concatenate((x, pulses['identity']))
        y = np.concatenate((y, pulses['identity']))
        z = np.concatenate((z, pulses['identity']))
    elif key is 'xp':
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
    return x, y, z


def operator_form_from_key(key, dim):
    g = qutip.basis(dim, 0)
    e = qutip.basis(dim, 1)
    sigma_x = g*e.dag() + e*g.dag()
    sigma_y = -1j * g*e.dag() + 1j * e*g.dag()
    # There's definitely some way of doing this more cleanly where I generate
    # lists of operators based on dimension by building things like a structure
    # hop[i][j] = |j><i|.
    if key is 'id':
        return qutip.qeye(dim)
    elif key is 'xp':
        return (-1j * np.pi / 2 * sigma_x).expm()
    elif key is 'yp':
        return (-1j * np.pi / 2 * sigma_y).expm()
    elif key is 'x2p':
        return (-1j * np.pi / 4 * sigma_x).expm()
    elif key is 'x2m':
        return (1j * np.pi / 4 * sigma_x).expm()
    elif key is 'y2p':
        return (-1j * np.pi / 4 * sigma_y).expm()
    elif key is 'y2m':
        return (1j * np.pi / 4 * sigma_y).expm()
    return None


def generate_simple_clifford_pulse_dictionary(pi_pulse_args,
                                              pi_half_pulse_args,
                                              sample_rate):
    """A method that constructs a pulse dictionary for id, xp, yp, x2p, x2m,
    y2p, and y2m pulses."""
    # Create the pi and pi/2 pulse definitions
    _, pi_xs, pi_ys, pi_zs = \
            pulse_utils.create_ge_envelopes(
                sample_rate,
                pi_pulse_args['tg'] * 2,
                pi_pulse_args)
                        
    _, pi_half_xs, pi_half_ys, pi_half_zs = \
            pulse_utils.create_ge_envelopes(
                sample_rate,
                pi_half_pulse_args['tg'] * 2,
                pi_half_pulse_args)

    pulse_dictionary = {
        'pi': pi_xs['r'],
        'pi_derivative': pi_ys['r'],
        'pi_detuning': pi_zs['r'],
        'pi_half': pi_half_xs['r'],
        'pi_half_derivative': pi_half_ys['r'],
        'pi_half_detuning': pi_half_zs['r'],
        'identity': np.zeros(0),
    }

    simple_clifford_indices = [0, 1, 2, 3, 12, 13, 14, 15, 16]
    keys = []
    for index in simple_clifford_indices:
        temp = []
        add_singleQ_clifford(index, temp)
        keys.append(temp[0])
    simple_clifford_dictionary = {}
    for key in keys:
        x = np.zeros(0)
        y = np.zeros(0)
        z = np.zeros(0)
        x, y, z = add_pulses_by_key(key, pulse_dictionary, x, y, z)
        simple_clifford_dictionary[key] = (np.copy(x), np.copy(y), np.copy(z))

    return simple_clifford_dictionary


def generate_simple_clifford_operator_dictionary(dim):
    """A method that constructs a dictionary of ideal operators representing
    id, xp, yp, x2p, x2m, y2p, y2m gates."""
    simple_clifford_indices = [0, 1, 2, 3, 12, 13, 14, 15, 16]
    keys = []
    for index in simple_clifford_indices:
        temp = []
        add_singleQ_clifford(index, temp)
        keys.append(temp[0])
    simple_clifford_dictionary = {}
    for key in keys:
        scratch = qutip.qeye(dim)
        simple_clifford_dictionary[key] = operator_form_from_key(key, dim) \
                                            * scratch
    return simple_clifford_dictionary


