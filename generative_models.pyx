from libc.math cimport log, log1p, M_PI, tan, sqrt, sin, pow, cos
from libc.stdlib cimport rand, RAND_MAX
from libc.stdio cimport printf
cimport cython
cimport numpy as np
import numpy as np


@cython.nonecheck(False)
@cython.cdivision(True)
cpdef double random_uniform():
    cdef double r = rand()
    return r / RAND_MAX


@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_exponential(double mu):

    cdef double u = random_uniform()
    return -mu * log1p(-u)


@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_gaussian():
    cdef double x1, x2, w

    w = 2.0
    while w >= 1.0:
        x1 = 2.0 * random_uniform() - 1.0
        x2 = 2.0 * random_uniform() - 1.0
        w = x1 * x1 + x2 * x2

    w = ((-2.0 * log(w)) / w) ** 0.5
    return x1 * w


@cython.nonecheck(False)
@cython.cdivision(True)
cdef double random_levy(double c, double alpha):
    cdef double u = M_PI * (random_uniform() - 0.5)
    cdef double v = 0.0
    cdef double t, s

    # Cauchy
    if alpha == 1.0:
        t = tan(u)
        return c * t

    while v == 0:
        v = random_exponential(1.0)

    # Gaussian
    if alpha == 2.0:
        t = 2 * sin(u) * sqrt(v)
        return c * t

    # General case
    t = sin(alpha * u) / pow(cos (u), 1 / alpha)
    s = pow(cos ((1 - alpha) * u) / v, (1 - alpha) / alpha)

    return c * t * s



@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
@cython.cdivision(True)
cdef double levy_lca_trial(double[:] p, double[:] x_acc, double kappa, double beta, double a,
              double ndt, double alpha, double s = 1.0, double dt = 0.001, int max_steps = 5000):
    """
    INPUT:
    res       - array to store (rt, resp)
    p         - drift rate
    x_acc     - activations (will usually be zeros)
    kappa     - leakage
    beta      - inhibition
    a         - threshold
    ndt       - non-decision time
    alpha     - heavy-tailness of distribution
    dt        - time step (0.001 = 1 ms)
    max_steps - maximum number of steps before terminating trial simulation
    """

    cdef int winner = 0
    cdef double n_steps = 0.0
    cdef int i
    cdef int n_acc = p.shape[0]
    cdef int resp
    cdef double max_p = 0.0
    cdef double rt
    cdef double[:] x = x_acc.copy()
    cdef double inhib_total
    cdef double rhs = sqrt(dt * s) # noise-damping factor
    cdef double c = 1. / sqrt(2)           # c levy parameter

    # Determine boundary of correct (incorrect will be -1)
    for i in range(n_acc):
        if p[i] >= max_p:
            max_p = p[i]
            resp = i

    # Start looping
    while winner == 0 and n_steps < max_steps:

        # Calculate total inhibition
        inhib_total = 0.0
        for i in range(n_acc):
            inhib_total += beta * x[i]

        # LCA equation
        for i in range(n_acc):
            x[i] = x[i] + (p[i] - kappa * x[i] - inhib_total + x[i] * beta) * dt + rhs * random_levy(c, alpha)
            x[i] = 0.0 if x[i] < 0.0 else x[i] # non-linearity

        # Increment step
        n_steps += 1.0

        # Check for winner
        for i in range(n_acc):
            if x[i] > a:
                winner = 1
                resp = 1 if resp == i else -1
                break

    cdef double max_x = 0.0
    cdef int max_acc
    if n_steps >= max_steps and winner == 0:        # just for the edge case that an accumulator exceeds a at max_steps
        # determine accumulator with highest evidence
        for i in range(n_acc):
            if x[i] > max_x:
                max_x = x[i]
                max_acc = i
        # determine if the accumulator with highest evidence is the "correct" one and set resp (= sign of returned RT)
        for i in range(n_acc):
            resp = 1 if max_acc == resp else -1


    # cdef double out =
    # printf("n_acc: %d, resp: %d, %f / %d steps, ndt=%f, rt=%f\n", n_acc, resp, n_steps, max_steps, ndt, out)  # debug
    return resp * (n_steps * dt + ndt)  # out


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
cpdef double[:] generate_levy_lca(int n_obs, double[:] p, double[:] x_acc, double kappa, double beta,
                 double a, double ndt, double alpha, double s = 1.0, double dt = 0.001, int max_steps = 5000):
    """
    Generate a dataset from the Levy-LCA evidence-accumulator model.
    ----------------
    INPUT:
    n_obs  - number of trials to generate
    p         - drift rate
    x_acc     - activations (will usually be zeros)
    kappa     - leakage
    beta      - inhibition
    a         - threshold
    ndt       - non-decision time
    alpha     - heavy-tailness of noise-distribution
    dt        - time step (0.001 = 1 ms)
    max_steps - maximum number of steps before terminating trial simulation

    ----------------
    OUTPUT:
    rts       - numpy array with reaction times
    """

    cdef int i
    cdef double[:] rts = np.zeros(n_obs, dtype=np.float)
    for i in range(n_obs):
        rts[i] = levy_lca_trial(p, x_acc, kappa, beta, a, ndt, alpha, s, dt, max_steps)
    return rts


@cython.boundscheck(False)
@cython.wraparound(False)
@cython.nonecheck(False)
def batch_generate_levy_lca(int n_obs, double[:, :] p, double[:, :] x_acc,
                            double[:] kappa, double[:] beta, double[:] a, double[:] ndt, double[:] alpha,
                            double s = 1.0, double dt = 0.001, int max_steps = 5000):
    """
    Generate a dataset from the Levy-LCA evidence-accumulator model.

    INPUT:
    n_sim     - number of datasets to generate
    n_obs     - number of trials to generate
    p         - drift rate
    x_acc     - activations (will usually be zeros)
    kappa     - leakage
    beta      - inhibition
    a         - threshold
    ndt       - non-decision time
    alpha     - heavy-tailness of noise-distribution
    dt        - time step (0.001 = 1 ms)
    max_steps - maximum number of steps before terminating trial simulation

    ----------------
    OUTPUT:
    rts       - numpy array with reaction times
    """

    cdef int i
    cdef int n_sim = p.shape[0]
    cdef double[:, :] sim_data = np.zeros((n_sim, n_obs), dtype=np.float)
    for i in range(n_sim):
        sim_data[i, :] = generate_levy_lca(n_obs, p[i, :], x_acc[i, :], kappa[i], beta[i], a[i], ndt[i], alpha[i], s, dt, max_steps)
    return sim_data
