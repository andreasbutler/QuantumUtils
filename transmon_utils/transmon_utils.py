import numpy as np

"""Reference: https://arxiv.org/pdf/cond-mat/0703002.pdf"""


def compute_josephson_energy_from_frequencies(ge_frequency, anharmonicity):
    """From w_ge = 1/2 * ( sqrt(8*E_J*E_C) ) - E_C"""
    charging_energy = -anharmonicity * 2 * np.pi
    josephson_energy = (2 * (ge_frequency + charging_energy)) ** 2 \
                        / (8 * charging_energy)
    return josephson_energy


def compute_charging_energy_from_frequencies(ge_frequency, anharmonicity):
    return -anharmonicity * 2 * np.pi


def compute_ge_frequency_from_energies(josephson_energy, charging_energy):
    """From w_ge = 1/2 * ( sqrt(8*E_J*E_C) ) - E_C"""
    w_ge = 1/2 * np.sqrt(8 * charging_energy * josephson_energy) \
                - charging_energy
    return w_ge / (2 * np.pi)


def compute_anharmonicity_from_energies(josephson_energy, charging_energy):
    return -charging_energy / (2 * np.pi)


def calculate_nth_energy_level(n, josephson_energy, charging_energy):
    if n < 0:
        raise Exception('Boooooo, no negative energy levels')
    if n != np.floor(n):
        raise Exception('Boooooo, no fractional energy levels')
    first_term = -josephson_energy
    second_term = np.sqrt(8 * charging_energy * josephson_energy) * (n + 1/2)
    third_term = -charging_energy / 12 * (6*n*n + 6*n + 3)


def calculate_nth_transition_energy(n, josephson_energy, charging_energy):
    """Defined as E_(n+1) - E_n"""
    if n < 0:
        raise Exception('Boooooo, no negative energy levels')
    if n != np.floor(n):
        raise Exception('Boooooo, no fractional energy levels')

    E_n_plus_1 = calculate_nth_energy_level(n + 1,
                    josephson_energy,
                    charging_energy)
    E_n = calculate_nth_energy_level(n, josephson_energy, charging_energy)

    return E_n_plus_1 - E_n


def calculate_nth_anharmonicity(n, josephson_energy, charging_energy):
    """Defined as E_(n+1)n - E_10"""
    E_10 = calculate_nth_transition_energy(0, 
                                        josephson_energy,
                                        charging_energy)
    E_n_plus_1_n = calculate_nth_transition_energy(n, 
                                        josephson_energy,
                                        charging_energy)
    return E_n_plus_1_n - E_10
