""" A super cool file for helping with DRAG stuff."""
import numpy as np
import scipy
from qutip import *
from qutip_utils import dsp_utils

# Empty envelopes
def empty_detuning_envelope(t, args):
    return 0


def empety_x_envelope(t, args):
    return 0


def empty_y_envelope(t, args):
    return 0


# Helper functions for building gaussian pulses
def gaussian(t, sigma):
    return np.exp(-t**2/(2*sigma**2))


def truncated_gaussian(t, sigma, t0, tn):
    erf_part = scipy.special.erf((tn)/(np.sqrt(2)*sigma))
    numerator = gaussian(t-t0, sigma) \
                    - gaussian(tn, sigma)
    denominator = np.sqrt(2*np.pi*sigma**2)*erf_part \
                      - 2*tn*gaussian(tn, sigma)
    return numerator/denominator


def truncated_gaussian_derivative(t, sigma, t0, tn):
    erf_part = scipy.special.erf((tn)/(np.sqrt(2)*sigma))
    numerator = -(t - t0)/(sigma**2) \
                    * gaussian(t-t0, sigma)
    denominator = np.sqrt(2*np.pi*sigma**2)*erf_part \
                  - 2*tn*gaussian(tn, sigma)
    return numerator/denominator


# Functions for bottom two level DRAG in an anharmonic oscillator
def x_envelope_ge(t, args):
    erf_part = scipy.special.erf(args['tn']/(np.sqrt(2)*args['tsigma']))
    numerator = np.exp(-(t - args['tg'])**2/(2*args['tsigma']**2)) - \
        np.exp(-args['tn']**2/(2*args['tsigma']**2))
    denominator = np.sqrt(2*np.pi*args['tsigma']**2)*erf_part - \
        2*args['tn']*np.exp(-args['tn']**2/(2*args['tsigma']**2))
    return args['x_coeff']*args['A']*numerator/denominator


def y_envelope_ge(t, args):
    erf_part = scipy.special.erf(args['tn']/(np.sqrt(2)*args['tsigma']))
    numerator = -(t-args['tg'])/(args['tsigma']**2) * \
        np.exp(-(t - args['tg'])**2/(2*args['tsigma']**2))
    denominator = np.sqrt(2*np.pi*args['tsigma']**2)*erf_part - \
        2*args['tn']*np.exp(-args['tn']**2/(2*args['tsigma']**2))
    return args['y_coeff']*args['A']*numerator/denominator


def det_envelope_ge(t, args):
    erf_part = scipy.special.erf(args['tn']/(np.sqrt(2)*args['tsigma']))
    numerator = np.exp(-(t - args['tg'])**2/(2*args['tsigma']**2)) - \
        np.exp(-args['tn']**2/(2*args['tsigma']**2))
    denominator = np.sqrt(2*np.pi*args['tsigma']**2)*erf_part - \
        2*args['tn']*np.exp(-args['tn']**2/(2*args['tsigma']**2))
    return args['det_coeff']*(args['A']*numerator/denominator)**2


# Functions for intermediate DRAG in an anharmonic oscillator
def x_envelope_ef(t, args):
    """In-Phase Quadrature Envelope for e->f DRAG."""
    return args['A']*truncated_gaussian(t, args['sigma'], args['t_g']/2, args['t_n']/2)


def y_envelope_ef(t, args):
    """Out-of-Phase Quadrature Envelope for e->f DRAG."""
    anharms = args['anharms']
    couplings = args['couplings']
    e = args['e']
    g = args['g']
    couplings = [c/g for c in couplings]
    coeff = -np.sqrt(couplings[e-1]**2 \
                     + (anharms[e+2]**2/anharms[e-1]**2) \
                     * couplings[e+1]**2) / (2*anharms[e+2])
    return args['A']*coeff*truncated_gaussian_derivative(t, args['sigma'], args['t_g']/2, args['t_n']/2)


def detuning_envelope_ef(t, args):
    """Detuning envelope for e->f DRAG."""
    anharms = args['anharms']
    couplings = args['couplings']
    e = args['e']
    g = args['g']
    couplings = [c/g for c in couplings]
    coeff = (couplings[e-1]**2 \
                - (anharms[e+2]**2/anharms[e-1]**2) \
                * couplings[e+1]) / (4*anharms[e+2])
    return coeff*(args['A']*truncated_gaussian(t, args['sigma'], args['t_g']/2, args['t_n']/2))**2


def save_envelopes(filename, times, x, y, det):
    """Utility for saving envelopes in triplets.
   
    Filtering is a costly operation, so to save time on simulations that
    require repeated use of the same filtered signals, it makes sense to
    save the envelopes for great reduction in simulation times.

    Args:
      filename: the path to the file where the envelopes should be saved
      x, y, det: the three DRAG envelopes. These must be numpy arrays and ought
          be of the same shape."""
    to_save = np.stack((times, x, y, det))
    np.savetxt(filename, to_save)


def read_envelopes(filename):
    """Utility for reading in envelopes saved as in save_envelope.

    Args:
      filename: the path to the save envelopes
    Returns:
      The time array and three DRAG envelopes as 1D numpy arrays."""
    stacked_envelopes = np.loadtxt(filename)
    return stacked_envelopes[0], stacked_envelopes[1], stacked_envelopes[2], stacked_envelopes[3]

def create_ge_envelopes(sample_rate,
                        gate_time,
                        envelope_args,
                        modulation_args=None,
                        quantization_args=None,
                        upsampling_args=None,
                        noise_args=None):
    xs, times = dsp_utils.create_custom_signal(
			      x_envelope_ge,
                              sample_rate,
                              gate_time,
	                      envelope_args=envelope_args,
 			      modulation_args=modulation_args,
                              quantization_args=quantization_args,
                              upsampling_args=upsampling_args,
                              noise_args=noise_args)
    ys, _ = dsp_utils.create_custom_signal(
			      y_envelope_ge,
                              sample_rate,
                              gate_time,
	                      envelope_args=envelope_args,
 			      modulation_args=modulation_args,
                              quantization_args=quantization_args,
                              upsampling_args=upsampling_args,
                              noise_args=noise_args)
    dets, _ = dsp_utils.create_custom_signal(
			      det_envelope_ge,
                              sample_rate,
                              gate_time,
	                      envelope_args=envelope_args,
 			      modulation_args=modulation_args,
                              quantization_args=quantization_args,
                              upsampling_args=upsampling_args,
                              noise_args=noise_args)
    return times, xs, ys, dets


def create_constant_detuning_DRAG_envelopes(sample_rate,
                                            gate_time,
                                            envelope_args,
                                            modulation_args=None,
                                            quantization_args=None,
                                            upsampling_args=None,
                                            noise_args=None):
    xs, times = dsp_utils.create_custom_signal(
			        x_envelope_ge,
                    sample_rate,
                    gate_time,
	                envelope_args=envelope_args,
 			        modulation_args=modulation_args,
                    quantization_args=quantization_args,
                    upsampling_args=upsampling_args,
                    noise_args=noise_args)
    ys, _ = dsp_utils.create_custom_signal(
			    y_envelope_ge,
                sample_rate,
                gate_time,
	            envelope_args=envelope_args,
 			    modulation_args=modulation_args,
                quantization_args=quantization_args,
                upsampling_args=upsampling_args,
                noise_args=noise_args)

    def const_function(t, args=None):
        return envelope_args['det_coeff']
    dets, _ = dsp_utils.create_custom_signal(
                  const_function,
                  sample_rate,
                  gate_time,
                  envelope_args=None,
                  modulation_args=None,
                  quantization_args=quantization_args,
                  upsampling_args=upsampling_args,
                  noise_args=noise_args)
    return times, xs, ys, dets


def generate_3LD_states_and_operators(dim, anharms, couplings):
    """ Create the QuTip states, operators, and hamiltonians for 3 level drag."""
    # States
    kets = [basis(dim, i) for i in range(dim)]
    projectors = [ket*ket.dag() for ket in kets]
    sigma_xs = [kets[i]*kets[i+1].dag() + kets[i+1]*kets[i].dag() for i in range(0,dim-1)]
    sigma_ys = [1j*(-1*kets[i]*kets[i+1].dag() + kets[i+1]*kets[i].dag()) for i in range(0,dim-1)]
    # Hamiltonian Parts
    H0 = sum(anharms[i]*projectors[i] for i in range(2, dim))
    Hz = sum(i*projectors[i] for i in range(1, dim))
    Hx = 1/2*(sum(couplings[i]*sigma_xs[i] for i in range(0, dim - 1)))
    Hy = 1/2*(sum(couplings[i]*sigma_ys[i] for i in range(0, dim - 1)))
    H = [H0, Hz, Hx, Hy]
    return kets, projectors, sigma_xs, sigma_ys, H
