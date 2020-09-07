# A really nice file of qutip utilities

from qutip import *
import matplotlib.pyplot as plt
import numpy as np
import scipy

def get_process_fidelity(H, t, input_states, target_gate, args, c_ops=None):
    target_states = [target_gate*input_state for input_state in input_states]
    process_fidelity = 0
    for input_state, target_state in zip(input_states, target_states):
        output = mesolve(H, input_state, t, args=args, c_ops=c_ops)
        final_state = output.states[-1]
        process_fidelity += fidelity(final_state, target_state)
    return process_fidelity/len(input_states)


def get_process_bloch_fidelity(H, t, target_gate, args, dim, g, e, c_ops=None):
    zero_ket = basis(dim, g)
    one_ket = basis(dim, e)
    xplus = (zero_ket + one_ket).unit()
    xminus = (zero_ket - one_ket).unit()
    yplus = (zero_ket + 1j*one_ket).unit()
    yminus = (zero_ket - 1j*one_ket).unit()
    input_states = [zero_ket, one_ket, xplus, xminus, yplus, yminus]
    return get_process_fidelity(H, t, input_states, target_gate, args, c_ops=c_ops)


def extract_state_populations(actual_states, target_states):
    pops = [[] for target in target_states]
    for actual_state in actual_states:
        for i in range(len(target_states)):
            pops[i].append(expect(target_states[i], actual_state))
    return pops


def extract_state_phases_over_time(states, reference, bases):
    phases = [[] for basis in bases]
    for state in states:
        reference_phase = np.angle(state[reference][0][0])
        for i in range(len(bases)):
            phases[i].append(np.angle(np.exp(-1j*reference_phase)*state[bases[i]][0][0]))
    return phases


def extract_qubit_info(states, level1, level2):
    xs = []
    ys = []
    zs = []
    pops_1 = []
    pops_2 = []
    for state in states:
        one_coeff = state[level1][0][0]
        two_coeff = state[level2][0][0]
        global_phase = np.angle(one_coeff)
        one_coeff = one_coeff*np.exp(-1j*global_phase)
        two_coeff = two_coeff*np.exp(-1j*global_phase)
        theta = 2*(np.arccos(np.real(one_coeff)))
        phi = np.angle(two_coeff)
        z = np.cos(theta)
        x = np.cos(phi)*np.sin(theta)
        y = np.sin(phi)*np.sin(theta)
        zs.append(z)
        xs.append(x)
        ys.append(y)
        pops_1.append(np.abs(one_coeff)**2)
        pops_2.append(np.abs(two_coeff)**2)
    return xs, ys, zs, pops_1, pops_2


def plot_bloch_profiles(xs, ys, zs):
    angles = np.linspace(0,2*np.pi,200)
    diameter = np.linspace(-1,1,200)
    zeros = np.linspace(0,0,200)

    plt.subplot(1,2,1)
    plt.scatter(xs,zs,marker='x')
    plt.plot(np.cos(angles), np.sin(angles), linewidth=1, color='black')
    plt.plot(zeros, diameter, linewidth=1, color='black')
    plt.plot(diameter, zeros, linewidth=1, color='black')
    plt.axis('equal')
    plt.xlabel('X')
    plt.ylabel('Z')

    plt.subplot(1,2,2)
    plt.scatter(ys,zs,marker='x',color='red')
    plt.plot(np.cos(angles), np.sin(angles), linewidth=1, color='black')
    plt.plot(zeros, diameter, linewidth=1, color='black')
    plt.plot(diameter, zeros, linewidth=1, color='black')
    plt.axis('equal')
    plt.xlabel('Y')

    plt.show()
