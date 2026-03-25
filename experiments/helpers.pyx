# distutils: language = c++
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def cholesky_rank1_update(np.ndarray[np.float64_t, ndim=2] R,
                          np.ndarray[np.float64_t, ndim=2] X_active,
                          np.ndarray[np.float64_t, ndim=1] x_new,
                          double eps=1e-12):
    cdef Py_ssize_t i, j, t = X_active.shape[1]
    cdef np.ndarray[np.float64_t, ndim=1] v = np.dot(X_active.T, x_new)
    cdef double s = np.dot(x_new, x_new)

    # Forward substitution: R.T z = v
    cdef np.ndarray[np.float64_t, ndim=1] z = np.zeros(t)
    for i in range(t):
        z[i] = v[i]
        for j in range(i):
            z[i] -= R[j, i] * z[j]
        z[i] /= R[i, i]

    cdef double r_squared = s - np.dot(z, z)
    if r_squared < eps:
        raise ValueError("Cholesky update failed: not positive definite")
    cdef double r = sqrt(r_squared)

    cdef np.ndarray[np.float64_t, ndim=2] R_new = np.zeros((t + 1, t + 1))
    for i in range(t):
        for j in range(i, t):
            R_new[i, j] = R[i, j]
        R_new[i, t] = z[i]
    R_new[t, t] = r
    return R_new


def compute_stepsize_gamma(np.ndarray[np.float64_t, ndim=1] corr,
                           np.ndarray[np.float64_t, ndim=1] a,
                           double A_active,
                           double C,
                           double tol,
                           np.ndarray[np.float64_t, ndim=1] gammas,
                           np.ndarray[np.int32_t, ndim=1] actives):
    """
    Compute candidate gammas for all variables; active variables get +inf.
    For inactive variables j, gamma_j = min^+{ (C - c_j)/(A - a_j), (C + c_j)/(A + a_j) }.
    """
    cdef int p = corr.shape[0]
    cdef bint is_active
    cdef int j, k
    cdef double gamma_plus, gamma_minus
    cdef double den1, den2

    for j in range(p):
        # check active
        is_active = False
        for k in range(actives.shape[0]):
            if actives[k] == j:
                is_active = True
                break

        if is_active:
            gammas[j] = np.inf
            continue

        # inactive: compute the two candidates, guarding denominators
        den1 = A_active - a[j]
        if den1 > tol:
            gamma_minus = (C - corr[j]) / den1
        else:
            gamma_minus = np.inf

        den2 = A_active + a[j]
        if den2 > tol:
            gamma_plus = (C + corr[j]) / den2
        else:
            gamma_plus = np.inf

        # positive-min selection
        if gamma_minus > tol and gamma_plus > tol:
            gammas[j] = gamma_minus if gamma_minus < gamma_plus else gamma_plus
        elif gamma_minus > tol:
            gammas[j] = gamma_minus
        elif gamma_plus > tol:
            gammas[j] = gamma_plus
        else:
            gammas[j] = np.inf

    return gammas