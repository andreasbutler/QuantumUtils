import qutip as qutip
import numpy as np

############### Utility methods for doing quantum state MLE reconstruction ###################

# I'm using Zradik Hradil's `Maximum Likelihood Methods for Quantum Mechanics.'
# (https://www.researchgate.net/publication/225572479_Maximum-Likelihood_Methods_in_Quantum_Mechanics)
# Specifically I'm using the algorithm described beginning on page 74.

def compute_R(rho, states_and_freqs):
    """ The method for computing the iterative update operator R

    Inputs:
        rho:                The current iterations estimate of rho
        states_and_freqs:   The POVM operators and the frequencies
                            with which they were measured."""
    # Make an empty container for holding R
    scratch = qutip.Qobj(np.zeros(rho.shape))

    # Iterate over POVM element - frequency pairs from simulated 'measurement'
    # results
    for state_and_freq in states_and_freqs:
        f = state_and_freq['f']
        state = state_and_freq['s']
        # Skip the states for which our current rho has no overlap (otherwise
        # we end up with a singularity in our equation due to dividing by 0)
        if qutip.expect(state, rho) == 0:
            continue

        # Otherwise add support on this POVM element to R with the prescribed
        # weight as per Hradil
        scratch += f*state/qutip.expect(state, rho)
    return scratch


def update(rho, R, alpha):
    """ The method for updating rho with R and alpha."""
    # Update rho with R using the 'dilution' paramter alpha, as per Hradil
    scratch = (1 - alpha)*rho + alpha/2 * (rho*R + R*rho)
    return scratch.unit()


##################### Test cases of MLE reconstruction #####################
def simple_2LS_artificial_sample_reconstruction():
    # A brute force simulation of a mixed state measurement. Reconstruction
    # should return a completely mixed state. It is uninteresting.
    num_plus_zs = 50
    num_minus_zs = 50
    num_plus_xs = 47
    num_minus_xs = 53
    num_plus_ys = 51
    num_minus_ys = 49

    num_experiments = 300

    plus_z = qutip.basis(2, 1)
    minus_z = qutip.basis(2, 0)

    plus_x = (plus_z + minus_z).unit()
    minus_x = (minus_z - plus_z).unit()
    
    plus_y = (1j*plus_z + minus_z).unit()
    minus_y = (minus_z - 1j*plus_z).unit()

    guess_list = [[0.5, 0], [0, 0.5]]
    guess = qutip.Qobj(np.matrix(guess_list))
    rho = guess.unit()

    sfpz = {'s': plus_z*(plus_z.dag()), 'f': num_plus_zs}
    sfmz = {'s': minus_z*(minus_z.dag()), 'f': num_minus_zs}
    sfpx = {'s': plus_x*(plus_x.dag()), 'f': num_plus_xs}
    sfmx = {'s': minus_x*(minus_x.dag()), 'f': num_minus_xs}
    sfpy = {'s': plus_y*(plus_y.dag()), 'f': num_plus_ys}
    sfmy = {'s': minus_y*(minus_y.dag()), 'f': num_minus_ys}

    states_and_freqs = [sfpz, sfmz, sfpx, sfmx, sfpy, sfmy]

    for i in range(100):
        R = compute_R(rho, states_and_freqs)
        rho = update(rho, R, 1)
    print(rho)


def sample_state_and_reconstruct():
    # This method constructs the |+y> state and uses it's measurement
    # statistics to generate a bunch of measurement results for 
    # different bases, and then reconstructs the state using mle
    # tomography and the measurement results.

    # Generate states to use in tomography
    ket_0 = qutip.basis(2, 0)
    ket_1 = qutip.basis(2, 1)
    ket_px = (ket_0 + ket_1).unit()
    ket_mx = (ket_0 - ket_1).unit()
    ket_py = (ket_0 + 1j*ket_1).unit()
    ket_my = (ket_0 - 1j*ket_1).unit()
    sx = qutip.sigmax()
    sy = qutip.sigmay()

    # Generate psi_0
    psi_0 = (ket_0 + 1j*ket_1).unit()
    rho = (ket_0*ket_0.dag() + ket_1*ket_1.dag()).dag()

    # Make density matrices of measurement states
    dm_mz = ket_0*(ket_0.dag())
    dm_pz = ket_1*(ket_1.dag())
    dm_mx = ket_mx*(ket_mx.dag())
    dm_px = ket_px*(ket_px.dag())
    dm_my = ket_my*(ket_my.dag())
    dm_py = ket_py*(ket_py.dag())

    # Simulate 100 measurements in each of the z, x, and y bases using simple
    # binomial samplings with the input state's associated measurement probs
    # in these bases.
    #
    # Here is where 'measurement noise' can be added if you'd like. You can
    # add noise by just explicitly distorting psi_0 or by changing the sampling
    # distribution.
    mz_p = qutip.expect(dm_mz, psi_0)
    fmz = np.random.binomial(100, mz_p)
    mx_p = qutip.expect(dm_mx, psi_0)
    fmx = np.random.binomial(100, mx_p)
    my_p = qutip.expect(dm_my, psi_0)
    fmy = np.random.binomial(100, my_p)

    print('Input state (Target):')
    print(psi_0 * psi_0.dag())
    print()

    print('Simulated frequencies for measuring state in: ')
    print('|-z>: ' + str(fmz))
    print('|-x>: ' + str(fmx))
    print('|-y>: ' + str(fmy))
    
    # Construct the input to the MLE reconstruction (POVM states and frequencies)
    sfmz = {'s': dm_mz, 'f': fmz}
    sfpz = {'s': dm_pz, 'f': 100 - fmz}
    sfmx = {'s': dm_mx, 'f': fmx}
    sfpx = {'s': dm_px, 'f': 100 - fmx}
    sfmy = {'s': dm_my, 'f': fmy}
    sfpy = {'s': dm_py, 'f': 100 - fmy}

    states_and_freqs = [sfmz, sfpz, sfmx, sfpx, sfmy, sfpy]

    # Reconstruct state
    rho = (dm_mz + dm_pz).unit()
    for i in range(1000):
        R = compute_R(rho, states_and_freqs)
        rho = update(rho, R, 1)
    print('Reconstructed density matrix:')
    print(rho)


def ThreeLS_artificial():
    # This test is less exciting than the two level system because I didn't use
    # a distribution plus state statistics to simulate measurements. All I did
    # was straight up plug the expectations of an input state on the different
    # bases into the frequencies used in the MLE state reconstruction. As you
    # can see though, it works. No reason you can't simulate sampling with these
    # probabilities to get something more interesting, I guess I just didn't
    # because I was lazy?

    # Boring setting up all the states stuff
    g = qutip.basis(3, 0)
    e = qutip.basis(3, 1)
    f = qutip.basis(3, 2)

    gg = g*(g.dag())
    ee = e*(e.dag())
    ff = f*(f.dag())

    ge = g*(e.dag())
    eg = e*(g.dag())
    gf = g*(f.dag())
    fg = f*(g.dag())
    ef = e*(f.dag())
    fe = f*(e.dag())

    ge_px = (g + e).unit()
    ge_px = ge_px*(ge_px.dag())
    ge_mx = (g - e).unit()
    ge_mx = ge_mx*(ge_mx.dag())

    ge_py = (g + 1j*e).unit()
    ge_py = ge_py*(ge_py.dag())
    ge_my = (g - 1j*e).unit()
    ge_my = ge_my*(ge_my.dag())

    gf_px = (g + f).unit()
    gf_px = gf_px*(gf_px.dag())
    gf_mx = (g - f).unit()
    gf_mx = gf_mx*(gf_mx.dag())

    gf_py = (g + 1j*f).unit()
    gf_py = gf_py*(gf_py.dag())
    gf_my = (g - 1j*f).unit()
    gf_my = gf_my*(gf_my.dag())

    ef_px = (e + f).unit()
    ef_px = ef_px*(ef_px.dag())
    ef_mx = (e - f).unit()
    ef_mx = ef_mx*(ef_mx.dag())

    ef_py = (e + 1j*f).unit()
    ef_py = ef_py*(ef_py.dag())
    ef_my = (e - 1j*f).unit()
    ef_my = ef_my*(ef_my.dag())

    # Define input (target state)
    psi_to_guess = (g + 2*e + 3*1j*f).unit()
    rho_to_guess = (psi_to_guess)*(psi_to_guess.dag())

    # Build POVM - frequency set used in MLE algorithm
    states_and_freqs = [
        {'s': gg, 'f':qutip.expect(rho_to_guess, gg)},
        {'s': ee, 'f':qutip.expect(rho_to_guess, ee)},
        {'s': ff, 'f':qutip.expect(rho_to_guess, ff)},
        {'s': ge_px, 'f':qutip.expect(rho_to_guess, ge_px)},
        {'s': ge_mx, 'f':qutip.expect(rho_to_guess, ge_mx)},
        {'s': ge_py, 'f':qutip.expect(rho_to_guess, ge_py)},
        {'s': ge_my, 'f':qutip.expect(rho_to_guess, ge_my)},
        {'s': gf_px, 'f':qutip.expect(rho_to_guess, gf_px)},
        {'s': gf_mx, 'f':qutip.expect(rho_to_guess, gf_mx)},
        {'s': gf_py, 'f':qutip.expect(rho_to_guess, gf_py)},
        {'s': gf_my, 'f':qutip.expect(rho_to_guess, gf_my)},
        {'s': ef_px, 'f':qutip.expect(rho_to_guess, ef_px)},
        {'s': ef_mx, 'f':qutip.expect(rho_to_guess, ef_mx)},
        {'s': ef_py, 'f':qutip.expect(rho_to_guess, ef_py)},
        {'s': ef_my, 'f':qutip.expect(rho_to_guess, ef_my)},
    ]

    # Input maximally mixed state as seed for MLE
    rho = qutip.Qobj(np.matrix([[1,0,0],[0,1,0],[0,0,1]]))
    for i in range(1000):
        R = compute_R(rho, states_and_freqs)
        rho = update(rho, R, 1)

    print('Input state (Target):')
    print(rho_to_guess)
    print()
    print('Reconstructed state:')
    print(rho.unit())
    print(np.linalg.eigvals(np.matrix(rho)))


if __name__ == "__main__":
    print()
    # TWO LEVEL SYSTEM EXAMPLE
    print('--------------- TWO LEVEL SYSTEM EXAMPLE ---------------')
    sample_state_and_reconstruct()
    print()
    print()

    # THREE LEVEL SYSTEM EXAMPLE
    print('--------------- THREE LEVEL SYSTEM EXAMPLE ---------------')
    ThreeLS_artificial()
