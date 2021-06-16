import numpy as np
import qutip as qutip
import time as time

from matplotlib import pyplot as plt

import sys
import os
sys.path.append('../../../../utils')

from tomography import coherent_state_tomography as cst
from tomography import coherent_state_tomography_copy as cst2
from visualization import state_visualization as state_vis
from quantum_utils import quantum_optics_utils as qou

from mpl_toolkits.mplot3d.axes3d import Axes3D
from math_utils import statistics_utils as su


if __name__ == "__main__":
    # Parameters - Make a small and coarse mesh initially because the outer
    # product will make the state space huge
    dim = 20
    mesh_size = 1/16
    max_x = 2
    
    # Generate a single coherent state povm
    povm, alphas, xs, ps = cst2.generate_coherent_state_POVM(max_x, 
                                                              mesh_size, 
                                                              dim,
                                                              cutoff=0)
    
    # Use the single coherent state povm to generate the joint two-photon povm
    s = time.time()
    print('Here we go...')
    two_photon_povm = []
    for povm_row_A in povm:
        rowi = []
        for povm_element_A in povm_row_A:
            rowj = []
            for povm_row_B in povm:
                rowk = []
                for povm_element_B in povm_row_A:
                    rowk.append(np.random.rand(dim*dim, dim*dim))
                rowj.append(rowk)
            rowi.append(rowj)
        two_photon_povm.append(rowi)
    e = time.time()
    print('Generating a 4-d POVM of ' + str(dim) + ' dimensional elements with a maximum ' \
            + 'quadrature value of ' + str(max_x) + ' and a mesh resolution ' \
            + 'of ' + str(mesh_size) + ' took ' + str(e - s) + ' s.')
