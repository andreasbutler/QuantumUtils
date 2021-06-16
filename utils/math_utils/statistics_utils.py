import numpy as np

"""A utility file for manipulating / sampling distributions."""

def sample_2d_distribution(distribution, samples, area_element=None):
    """Samples a 2d histogram from a given 2d distribution.

    Assumes `distribution` is a 2D np array"""
    rows = distribution.shape[0]
    cols = distribution.shape[1]
    flattened_distribution = distribution.flatten()
    # Normalize just in case
    extended_distribution = np.zeros(rows*cols + 1)
    extended_distribution[:rows*cols] = flattened_distribution
    if area_element is not None:
        extended_distribution[-1] = 1/area_element - np.sum(flattened_distribution)#*area_element
        if np.abs(extended_distribution[-1]) < 1e-8:
            extended_distribution[-1] = 0
    else:
        extended_distribution[-1] = 1 - np.sum(flattened_distribution)
    extended_distribution /= np.sum(extended_distribution)

    indices = np.arange(rows*cols + 1)
    
    data = np.random.choice(indices, p=extended_distribution, size=samples)
    hist = np.histogram(data, bins=rows*cols, range=(0, rows*cols - 1))[0]
    hist = hist[:rows*cols]
    hist = np.reshape(hist, (rows, cols))
    return hist / np.sum(hist)


def sample_2d_distribution_values(distribution, values, samples, area_element=None):
    rows = distribution.shape[0]
    cols = distribution.shape[1]
    flattened_distribution = distribution.flatten()
    # Normalize just in case
    extended_distribution = np.zeros(rows*cols + 1)
    extended_distribution[:rows*cols] = flattened_distribution
    if area_element is not None:
        extended_distribution[-1] = 1/area_element - np.sum(flattened_distribution)#*area_element
        if np.abs(extended_distribution[-1]) < 1e-8:
            extended_distribution[-1] = 0
    else:
        extended_distribution[-1] = 1 - np.sum(flattened_distribution)
    extended_distribution /= np.sum(extended_distribution)

    values = values.flatten()
    extended_values = np.zeros(rows*cols + 1, dtype=np.complex128)
    extended_values[:rows*cols] = values

    data = np.random.choice(extended_values, p=extended_distribution, size=samples)
    return np.squeeze(data)


def evenly_histogram_2d_data(x, y, xs, ps, mesh_size):
    r = [[xs[0] - mesh_size/2, xs[-1] + mesh_size/2], [ps[0] - mesh_size/2, ps[-1] + mesh_size/2]]
    return np.histogram2d(x,
                          y, 
                          range=r,
                          bins = len(xs),
                          density = True)[0]


def average_over_histogram(histogram, rv_values, generalized_area):
    return generalized_area * np.sum(histogram * rv_values)
