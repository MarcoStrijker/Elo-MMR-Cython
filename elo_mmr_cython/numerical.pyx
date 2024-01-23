
from libc.math cimport cosh, tanh, atanh, exp, erfc
from libc.math cimport M_SQRT2, M_2_SQRTPI, M_PI


import warnings


cdef public double TANH_MULTIPLIER = M_PI / 1.7320508075688772


cpdef double standard_logistic_pdf(double z):
    return cosh(0.25 * TANH_MULTIPLIER * z) ** (-2) * 0.25 * TANH_MULTIPLIER


cpdef double standard_logistic_cdf(double z):
    return 0.5 + 0.5 * tanh(0.5 * TANH_MULTIPLIER * z)


cdef double standard_logistic_cdf_inv(double prob):
    return atanh(2 * prob - 1) * 2 / TANH_MULTIPLIER


cpdef double standard_normal_pdf(double z):
    cdef double NORMALIZE = 0.5 * M_2_SQRTPI / M_SQRT2
    return NORMALIZE * exp(-0.5 * z * z)
    

cpdef double standard_normal_cdf(double z):
    # TODO: check if erf is right, or if we need to use erfc
    return 0.5 * erfc(z / M_SQRT2)


# cpdef double standard_normal_cdf_inv(double prob):
#     return -M_SQRT2 * erfc_inv(2 * prob)


cdef double clamp(double x, double low, double high):
    """Clamps a value between two bounds

    Arguments:
        x {double} -- value to clamp
        low {double} -- lower bound
        high {double} -- upper bound

    Returns:
        double -- clamped value
    """
    return max(low, min(x, high))


cdef public double recip(double x):
    """Returns the reciprocal of a number

    Arguments:
        x {double} -- number to find the reciprocal of

    Returns:
        double -- reciprocal of x
    """
    return <double>(1 / x)



cdef float solve_newton(tuple bounds, f):
    """ Returns the root of a function using Newton's method

    Arguments:
        bounds {tuple} -- bounds of the root
        f {function} -- function to find the root of

    Returns:
        float -- root of the function
    """
    cdef double low = bounds[0]
    cdef double high = bounds[1]
    cdef double guess = (bounds[0] + bounds[1]) * 0.5
    cdef double val, val_prime, extrapolate

    while True:
        sum, sum_prime = f(guess)
        extrapolate = guess - sum / sum_prime

        if extrapolate <= bounds[0]:
            low = guess
            guess = clamp(extrapolate, high - 0.75 * (high - low), high)
        else:
            high = guess
            guess = clamp(extrapolate, low, low + 0.75 * (high - low))

        valid_range = (low >= guess or guess >= high)
        if not valid_range:
            continue

        if abs(sum) < 1e-10:
            warnings.warn("Possible failure to converge @ {guess}: s={sum}, s'={sum_prime}")

        return guess
