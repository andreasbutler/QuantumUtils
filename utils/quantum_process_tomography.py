import numpy as np
import qutip as qutip
from qutip_utils import pulse_utils as pulse_utils
from qutip_utils import dsp_utils as dsp_utils

def generate_clifford_gate_dictionary(pi_pulse_parameters,
                                      pi_half_pulse_parameters,
                                      dim=2):
    if dim != 2:
        raise Exception('Lol, dream on bud')


def qpt_concatenate(init, proc, meas):
    """A simple utility method for concatenating the separate X, Y, and Z
    pulses of the different sections of a qpt experiment."""
    x = np.concatenate((init[0], proc[0], meas[0]))
    y = np.concatenate((init[1], proc[1], meas[1]))
    z = np.concatenate((init[2], proc[2], meas[2]))
    return (x, y, z)


##### UTILITY FUNCTIONS FOR MLE RECONSTRUCTION ################################
def compute_p_mls(E, input_states, measured_states):
    p_mls = np.zeros((len(input_states), len(measured_states)))
    for m, input_state in enumerate(input_states):
        for l, measured_state in enumerate(measured_states):
            rho_tensor_proj = qutip.tensor(input_state.trans(),
                                                 measured_state)
            E_rho_tensor_proj = E * rho_tensor_proj
            p_ml = np.real(np.trace(E_rho_tensor_proj))
            p_mls[m][l] = p_ml
    return p_mls
            

def compute_K(f_mls,
              p_mls,
              input_states,
              measured_states):
    K_parts = []
    for m, input_state in enumerate(input_states):
        for l, measured_state in enumerate(measured_states):
            ratio = f_mls[m][l] / p_mls[m][l]
            operator = qutip.tensor(input_state.trans(), measured_state)
            K_parts.append(ratio * operator)
    return sum(K_parts)


def compute_Lambda(K, E):
    KEK = K * E * K
    little_lambda_squared = KEK.ptrace(1)
    little_lambda = little_lambda_squared.sqrtm()
    little_lambda_inv = qutip.Qobj(np.linalg.inv(little_lambda))
    Lambda = qutip.tensor(little_lambda_inv, qutip.qeye(little_lambda.shape[-1]))
    return Lambda


def mle_process_reconstruction(input_states,
                               measured_states,
                               measured_frequencies):
    E_0 = qutip.tensor(qutip.qeye(2), qutip.qeye(2)) / (2)
    for i in range(100):
        p_mls = compute_p_mls(E_0, input_states, measured_states)
        K = compute_K(measured_frequencies,
                      p_mls,
                      input_states,
                      measured_states)
        Lambda = compute_Lambda(K, E_0)
        E_0 = Lambda * K * E_0 * K * Lambda
    return E_0


###############################################################################
    

def simulate_qpt(preparation_gates,
                 measurement_gates,
                 process_gates,
                 gate_dictionary,
                 simulation_parameters,
                 ideal_initial_states=None,
                 ideal_povm_states=None,
                 reconstruction_method='MLE',
                 result_representation='CHOI'):
    """A method for running a transmon QPT simulation.

    This method assumes the transmon Hamiltonian and performs QPT for a set of
    process pulses and sets of SPAM gates. By default it returns the Choi
    matrices of the processes reconstructed using maximum-likelihood
    estimation.

    Args:
        preparation_gates: A set of labels serving as keys into the
            gate_dictionary that define the set of gates used for state
            initialization.
        measurement_gates: A set of labels serving as keys into the
            gate_dictionary that define the set of gates used for rotating
            measurement bases.
        process_gates: A set of labels serving as keys into the gate_dictionary
            that define the subjects of the tomography experiments.
        gate_dictionary: A dictionary mapping gate labels to numpy arrays
            defining the drive pulse shapes assuming the typical transmon
            Hamiltonian. Must have a key for every unique entry in the union of
            preparation_gates, measurement_gates, and process_gates. The keys
            should be strings and the elements should be triples of numpy arrays
            defining the X, Y, and Z pulses associated with the gate labeled by
            the key.
        ideal_initial_states: The density matrices (Qobj) representing the ideal
            results of applying the initialization pulses. If these are `None`
            then the assumption is that all the preparation gates are Cliffords,
            the input state to initialization is ground, and the labels in
            preparation gates adhere to the convention in the method
            `clifford_labels_to_prepared_states`.
        ideal_povm_states: The density matrices (Qobj) representing the ideal
            basis elements of the measurements. If these are `None`
            then the assumption is that all the preparation gates are Cliffords,
            the input state to initialization is ground, and the labels in
            preparation gates adhere to the convention in the method
            `clifford_labels_to_prepared_states`.
        reconstruction_method: The process matrix reconstruction algorithm.
        result_representation: The process matrix representation.
    Returns:
        A list of process matrices of the representation defined by
        `result_representation`
    """
    sampling_rate = simulation_parameters['sampling_rate']
    H_parts = simulation_parameters['H']
    psi_0 = simulation_parameters['psi_0']
    e_ops = simulation_parameters['e_ops']
    c_ops = simulation_parameters['c_ops']
    process_populations = []
    for process in process_gates:
        populations = []
        for initialization in preparation_gates:
            initialization_populations = []
            for measurement_basis_transform in measurement_gates:                 
                measurement_basis_populations = []
                (x, y, z) = qpt_concatenate(gate_dictionary[initialization],
                                gate_dictionary[process],
                                gate_dictionary[measurement_basis_transform])
                times = dsp_utils.get_times(x, sampling_rate)


                # Right now there should only ever be one element in e_ops
                # because to get the separate populations say of the |g> and
                # |e> states in the Z basis rather than extract both at once
                # I run two separate experiments with the same initialization
                # and process but in one map |g> -> |g> before measurement and
                # in the other map |g> -> |e>, (in other words I'm only ever
                # 'measuring' ground state population, and thus overworking by
                # simulating redundant measurement bases that differ only by
                # permuting the computational basis)
                #
                # Originally I was not doing this, and perhaps to save
                # simulation time at some point it ought to change, but for now
                # this makes the assumptions about system dimensionality and
                # what states we're considering much simpler.
                H = [H_parts[0], [H_parts[1], x], [H_parts[2], y], [H_parts[3], z]]
                results = qutip.mesolve(H,
                            psi_0,
                            times,
                            c_ops=c_ops,
                            e_ops=e_ops)
                output = qutip.mesolve(H, psi_0, times)
                for population in results.expect:
                    initialization_populations.append(population[-1])
            populations.append(initialization_populations)
        process_populations.append(populations)

    # Do reconstruction
    matrices = []
    for qpt_result in process_populations:
        if reconstruction_method == 'MLE':
            process_choi_matrix = mle_process_reconstruction(
                                      ideal_initial_states,
                                      ideal_povm_states,
                                      qpt_result)
            if result_representation == 'CHOI':
                matrices.append(process_choi_matrix)
            else:
                raise Exception('Not yet for that represenation...')
        else:
            raise Exception('Not quite yet sadly...')
    return matrices
     
