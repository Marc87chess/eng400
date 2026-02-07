from control import lqr
import numpy as np

def build_tan_pwl(theta_max_rad=np.deg2rad(30), n_bins=30):
    # Bin edges from -theta_max to +theta_max
    edges = np.linspace(-theta_max_rad, theta_max_rad, n_bins + 1)

    a = np.zeros(n_bins)  # slopes
    b = np.zeros(n_bins)  # intercepts

    # Secant line on each bin: sin(t) ~ a_i * t + b_i
    for i in range(n_bins):
        t0, t1 = edges[i], edges[i+1]
        s0, s1 = np.tan(t0), np.tan(t1)
        a[i] = (s1 - s0) / (t1 - t0)
        b[i] = s0 - a[i] * t0

    return edges, a, b

def bin_index(angle, edges):
    # clamp angle into range
    angle = np.clip(angle, edges[0], edges[-1] - 1e-12)
    # index i such that edges[i] <= angle < edges[i+1]
    i = int(np.searchsorted(edges, angle, side="right") - 1)
    return max(0, min(i, len(edges) - 2))


def make_deviation_model(A, B, c, x0=None):
    """
    Takes one packed block (12x17) containing [A | B | c] and returns a
    deviation-coordinate model that has NO constant term:

        dx_dot = A dx + B du

    where dx = x - x0 and du = u - u0, and (x0,u0) satisfy:
        0 = A x0 + B u0 + c

    Inputs:
      block : (12,17) array from bank[ip,it]
      x0    : optional (12,) chosen operating state. If None, uses zeros.

    Returns:
      A, B          : same matrices (views/copies depending on slicing)
      x0, u0        : trim point
      rhs           : the required steady-state B*u0 = -(A*x0 + c) (debug)
    """
   

    if x0 is None:
        x0 = np.zeros(12)

    # Solve for u0 so that xdot = 0 at (x0,u0):
    # 0 = A x0 + B u0 + c  ->  B u0 = -(A x0 + c)
    rhs = -(A @ x0 + c)
    u0, *_ = np.linalg.lstsq(B, rhs, rcond=None)

    return A, B, x0, u0, rhs

def to_deviation_coords(x, u, x0, u0):
    """Convert absolute (x,u) to deviation (dx,du)."""
    return x - x0, u - u0

def deviation_dynamics(A, B, dx, du):
    """Compute dx_dot in deviation coordinates."""
    return A @ dx + B @ du
def make_all_devation_models(bank,Nphi=30, Ntheta=30):
    bank_a = np.zeros((Nphi, Ntheta, 12, 12))
    bank_b = np.zeros((Nphi, Ntheta, 12, 4))
    """
    Converts all blocks in the bank to deviation-coordinate models.

    Inputs:
      bank : (Nphi, Ntheta, 12, 17) array of packed blocks

    Returns:
      bank_a, bank_b : (Nphi, Ntheta, 12, 12) and (Nphi, Ntheta, 12, 4) arrays of deviation matrices
      dev_bank : (Nphi, Ntheta) array of tuples (x0, u0)
    """
    Nphi, Ntheta, _, _ = bank.shape
    dev_bank = np.empty((Nphi, Ntheta), dtype=object)

    for ip in range(Nphi):
        for it in range(Ntheta):
            block = bank[ip, it]
            A, B, x0, u0, _ = make_deviation_model(block[:, 0:12], block[:, 12:16], block[:, 16])
            dev_bank[ip, it] = (x0, u0)
            bank_a[ip, it] = A
            bank_b[ip, it] = B

    return bank_a, bank_b, dev_bank
