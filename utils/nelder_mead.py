# A utility file for performing Nelder-Mead optimization in arbitrary
# dimensional space.
import numpy as np

def eval_centroid(evaluated_simplex):
    """ A function for evaluating the centroid of the d-1 best points of the
    simplex."""
    centroid = np.zeros(len(evaluated_simplex[0][0]))
    for p in evaluated_simplex[:-1]:
        centroid += p[0]
    return centroid / len(evaluated_simplex[0][0])


# Define deformation functions
def reflected_point(worst_point, centroid, reflection_coefficient):
    """Reflection:
    Evaluate the point that is the reflection of the 'worst' point
    (the point with the highest cost function value, and so the last point in
    the simplex) across the centroid of the simplex."""
    return centroid + reflection_coefficient*(centroid - worst_point)


def expand_reflected_point(reflected_point, centroid, expansion_coefficient):
    """Expansion:
    If the point we reflect is literally the best point in the whole universe
    that we know of, baby we're going all in, expand that mfr even more in the
    same direction."""
    return centroid + expansion_coefficient*(reflected_point - centroid)


def contract_worst_point(worst_point, centroid, contraction_coefficient):
    """ Contraction:
    If the point we reflect is worse than the second to last point, we want to
    step into the simplex instead, so we take the worst point and put it some way
    between itself and the centroid."""
    return centroid + contraction_coefficient*(worst_point - centroid)


def shrink(evaluated_simplex, shrink_coefficient, cost_function=None):
    """Shrink:
    If all of the above are bust, slurp everything in towards the best point."""
    if cost_function is None:
        raise Error('Error. No cost function provided.')
    best_point = evaluated_simplex[0][0]
    for i in range(1, len(evaluated_simplex)):
        point_i = evaluated_simplex[i][0]
        update_point = best_point + shrink_coefficient*(point_i - best_point)
        evaluated_simplex[i] = [update_point, cost_function(update_point)]
    evaluated_simplex = sorted(evaluated_simplex, key=lambda point: point[1])


def run_nelder_mead(evaluated_simplex, cost_function, nm_coeffs, max_iterations=1e6):
    """ The actual algorithm. """
    iterations = -1
    centroids = []
    best_values = []
    worst_values = []
    while True:
        iterations += 1
        
        # Sort our simplex vertices by their cost function values
        evaluated_simplex = sorted(evaluated_simplex, key=lambda point: point[1])
        
        # Evaluate the centroid of the n-1 best vertices
        centroid = eval_centroid(evaluated_simplex)
        centroids.append(centroid)

        # Grab the worst point and worst value
        wp = evaluated_simplex[-1][0]
        wv = evaluated_simplex[-1][1]
        # Grab the second-to-worst value
        swv = evaluated_simplex[-2][1]
        # Grab the best value and best point
        bp = evaluated_simplex[0][0]
        bv = evaluated_simplex[0][1]

        best_values.append(bv)
        worst_values.append(wv)
        
        # terminate if this is seeming interminable
        if (iterations > max_iterations):
            break
        
        # terminate if error is small and stuff
        if bv < 1e-8:
            break

        rp = reflected_point(wp, centroid, nm_coeffs['r'])
        rv = cost_function(rp)
        if rv < swv and rv > bv:
            evaluated_simplex[-1] = [rp, rv]
            continue

        if rv < bv:
            ep = expand_reflected_point(rp, centroid, nm_coeffs['e'])
            ev = cost_function(ep)
            if ev < rv:
                evaluated_simplex[-1] = [ep, ev]
            else:
                evaluated_simplex[-1] = [rp, rv]
            continue
        
        cp = contract_worst_point(wp, centroid, nm_coeffs['c'])
        cv = cost_function(cp)
        if cv < wv:
            evaluated_simplex[-1] = [cp, cv]
            continue
        
        shrink(evaluated_simplex, nm_coeffs['s'], cost_function)
    
    return bp, bv, centroids, best_values, worst_values
