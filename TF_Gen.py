import numpy as np
from matrix_gen import *
from scipy.signal import ss2tf


def generate_tf_bank_from_deviation(bank_a, bank_b, C, D):
    """
    Generate transfer functions from deviation-coordinate models.

    Inputs:
      bank_a : (Nphi, Ntheta, 12, 12) deviation A matrices
      bank_b : (Nphi, Ntheta, 12, 4)  deviation B matrices
      C      : (ny, 12) output matrix
      D      : (ny, 4)  feedthrough matrix

    Returns:
      tf_bank : tf_bank[ip][it][u] -> {"num": num, "den": den}
    """
    Nphi, Ntheta = bank_a.shape[0], bank_a.shape[1]
    n_inputs = bank_b.shape[3]

    tf_bank = [[None for _ in range(Ntheta)] for _ in range(Nphi)]

    for ip in range(Nphi):
        for it in range(Ntheta):
            A = bank_a[ip, it]
            B = bank_b[ip, it]

            tf_inputs = []
            for u in range(n_inputs):
                num, den = ss2tf(A, B, C, D, input=u)
                tf_inputs.append({"num": num, "den": den})

            tf_bank[ip][it] = tf_inputs

    return tf_bank


# Example usage:
edges, a, b = build_sin_pwl(theta_max_rad=np.deg2rad(35), n_bins=30)
