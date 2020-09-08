import numpy as np
import qutip as qutip

import transmon_utils as transmon_utils

class Transmon:
    """TODO: Write a good description of this classy boy."""
    def __init__(self,
                 ge_frequency=None,
                 anharmonicity=None,
                 josephson_energy=None,
                 charging_energy=None):
        """This initializer should not be invoked except through 
        pseudoconstructor classmethods like the transmon_from_* methods defined
        below. It exists essentially as a switch statement for various transmon
        parameterizations."""
        if (ge_frequency is not None \
                or anharmonicity is not None) \
            and (josephson_energy is not None \
                or charging_energy is not None):
            raise Exception('You may only instantiate transmon with one \
                             complete parameter set.')

        if ge_frequency is not None and anharmonicity is not None:
            self.ge_frequency = ge_frequency
            self.anharmonicity = anharmonicity
            self.ef_frequency = ge_frequency + anharmonicity
 
            self.josephson_energy = \
                transmon_utils.compute_josephson_energy_from_frequencies(
                    ge_frequency,
                    anharmonicity)
 
            self.charging_energy = \
                transmon_utils.compute_charging_energy_from_frequencies(
                    ge_frequency,
                    anharmonicity)

        elif josephson_energy is not None and charging_energy is not None:
            self.josephson_energy = josephson_energy
            self.charging_energy = charging_energy

            self.ge_frequency = transmon_utils.ge_frequency_from_energies(
                                    josephson_energy,
                                    charging_energy)

            self.anharmonicity = transmon_utils.anharmonicity_from_energies(
                                    josephson_energy,
                                    charging_energy)

            self.ef_frequency = self.ge_frequency + anharmonicity

        else:
            raise Exception('Incomplete parameterization')


    @classmethod
    def transmon_from_frequency_parameterization(cls,
                                                 ge_frequency,
                                                 anharmonicity):
        return cls(ge_frequency=ge_frequency, anharmonicity=anharmonicity)


    @classmethod
    def transmon_from_energy_parameterization(cls,
                                              josephson_energy,
                                              charging_energy):
        return cls(josephson_energy=josephson_energy,
                   charging_energy=charging_energy)


    def get_rotating_frame_hamiltonian(self, dim):
        anharmonicities = [transmon_utils.calculate_nth_anharmonicity(n,
                                self.josephson_energy,
                                self.charging_energy) \
                            for n in range(1, dim)]
        couplings = [np.sqrt(n) for n in range(1, dim)]
    
    def get_lab_frame_hamiltonian(self, dim):
        w_ge = 2 * np.pi * self.ge_frequency
