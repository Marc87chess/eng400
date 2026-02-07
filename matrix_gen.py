from control import lqr
import numpy as np

# Parameters
g  = 9.81   # gravity
m  = 1.0    # mass
Ix = 0.01   # inertia about body x
Iy = 0.01   # inertia about body y
Iz = 0.02   # inertia about body z

# State: [x, y, z, xdot, ydot, zdot, phi, theta, psi, phidot, thetadot, psidot]
# Input: [u1 (thrust), u2 (tau_x), u3 (tau_y), u4 (tau_z)]

A = np.zeros((12, 12))
B = np.zeros((12, 4))

# Kinematics: position derivatives
A[0, 3] = 1.0   # x' = xdot
A[1, 4] = 1.0   # y' = ydot
A[2, 5] = 1.0   # z' = zdot

# Translational dynamics (small-angle around hover)
A[3, 7] = -g    # xddot = -g * theta
A[4, 6] =  g    # yddot =  g * phi
# zddot = -(u1/m) -> no state coupling in A

B[5, 0] = -1.0 / m   # zddot input from thrust u1

# Kinematics: angles derivatives
A[6,  9] = 1.0  # phi'   = phidot
A[7, 10] = 1.0  # theta' = thetadot
A[8, 11] = 1.0  # psi'   = psidot

# Rotational dynamics
B[9,  1] = 1.0 / Ix   # phiddot   = u2 / Ix
B[10, 2] = 1.0 / Iy   # thetaddot = u3 / Iy
B[11, 3] = 1.0 / Iz   # psiddot   = u4 / Iz

# Optional: full-state output
C = np.eye(12)
D = np.zeros((12, 4))

print("A=\n", A)
print("B=\n", B)





###########C MATRIX

# Output: [vx, vy, vz, phi, theta, psi]
C = np.zeros((6, 12))

# velocities
C[0, 3] = 1.0   # vx = xdot
C[1, 4] = 1.0   # vy = ydot
C[2, 5] = 1.0   # vz = zdot

# angles
C[3, 6] = 1.0   # phi
C[4, 7] = 1.0   # theta
C[5, 8] = 1.0   # psi

# No direct feedthrough
D = np.zeros((6, 4))

print("C=\n", C)





def build_sin_pwl(theta_max_rad=np.deg2rad(30), n_bins=30):
    # Bin edges from -theta_max to +theta_max
    edges = np.linspace(-theta_max_rad, theta_max_rad, n_bins + 1)

    a = np.zeros(n_bins)  # slopes
    b = np.zeros(n_bins)  # intercepts

    # Secant line on each bin: sin(t) ~ a_i * t + b_i
    for i in range(n_bins):
        t0, t1 = edges[i], edges[i+1]
        s0, s1 = np.sin(t0), np.sin(t1)
        a[i] = (s1 - s0) / (t1 - t0)
        b[i] = s0 - a[i] * t0

    return edges, a, b

def bin_index(angle, edges):
    # clamp angle into range
    angle = np.clip(angle, edges[0], edges[-1] - 1e-12)
    # index i such that edges[i] <= angle < edges[i+1]
    i = int(np.searchsorted(edges, angle, side="right") - 1)
    return max(0, min(i, len(edges) - 2))

def precompute_bank_2d(Nphi=30, Ntheta=30,
                       phi_max=np.deg2rad(35), theta_max=np.deg2rad(35),
                       g=9.81, m=2.0, Ix=0.2, Iy=0.2, Iz=0.2):
    # Separate tables for phi and theta
    edges_phi, a_phi, b_phi = build_sin_pwl(phi_max, Nphi)
    edges_th,  a_th,  b_th  = build_sin_pwl(theta_max, Ntheta)

    # One packed tensor: bank[i_phi, i_theta] is a (12 x 17) block
    # columns: [A(12 cols) | B(4 cols) | c(1 col)]
    bank = np.zeros((Nphi, Ntheta, 12, 17))

    for ip in range(Nphi):
        for it in range(Ntheta):
            A = np.zeros((12, 12))
            B = np.zeros((12, 4))
            c = np.zeros((12,))

            # Kinematics
            A[0,3]=1; A[1,4]=1; A[2,5]=1
            A[6,9]=1; A[7,10]=1; A[8,11]=1
            b = -0.1 # drag coefficient
            A[3,3] = b; A[4,4]=b; A[5,5]=b #wind
            b1 = -0.1 # drag coefficient
            A[9,9]=b1; A[10,10]=b1; A[11,11]=b1 #spining drag
            # Inputs (choose sign convention you want)
            B[5,0]  =  1.0/m
            B[4,0]  =  (a_phi[ip]/m)
            B[3,0]  =  -(a_th[it]/m)
            B[9,1]  =  1.0/Ix
            B[10,2] =  1.0/Iy
            B[11,3] =  1.0/Iz

            # Scheduled PWL sin
            # xddot = -g*sin(theta)
            A[3,7] = -g * a_th[it]
            c[3]   = -g * b_th[it] 

            # yddot =  g*sin(phi)
            A[4,6] =  g * a_phi[ip]
            c[4]   =  g * b_phi[ip]

            # Pack into one block
            block = bank[ip, it]
            block[:, 0:12]  = A
            block[:, 12:16] = B
            block[:, 16]    = c

    return bank, edges_phi, edges_th

def unpack_block(block):
    A = block[:, 0:12]
    B = block[:, 12:16]
    c = block[:, 16]
    return A, B, c



# -----------------------------------------------------------------------------
# WHY WE MUST REMOVE THE CONSTANT TERM (c) BEFORE COMPUTING TRANSFER FUNCTIONS
#
# Our per-bin dynamics have the AFFINE form:
#
#     xdot = A_i x + B u + c_i
#
# A transfer function is defined ONLY for LTI systems of the form:
#
#     xdot = A x + B u
#     y    = C x + D u
#
# because the transfer function G(s) = Y(s)/U(s) assumes:
#   - linearity
#   - time invariance
#   - zero initial conditions
#   - NO standalone forcing term
#
# If c_i is present, then in Laplace space:
#
#     X(s) = (sI - A_i)^(-1) B U(s) + (sI - A_i)^(-1) * (c_i / s)
#
# The second term is independent of U(s), so Y(s)/U(s) is NOT defined.
# In other words: constant forces (gravity, trim bias, tilt bias) are NOT
# part of the input-output dynamics that transfer functions describe.
#
# -------------------------------------------------------------------------
# CORRECT FIX: SHIFT TO DEVIATION VARIABLES (THIS IS EXACT, NOT A HACK)
#
# For each bin i, choose an equilibrium (x0_i, u0_i) such that:
#
#     0 = A_i x0_i + B u0_i + c_i
#
# Define deviations:
#
#     dx = x - x0_i
#     du = u - u0_i
#
# Substituting into the full dynamics:
#
#     d(dx)/dt = A_i dx + B du
#
# The constant term disappears EXACTLY because it is canceled by the
# equilibrium input u0_i. No approximation is introduced here.
#
# -------------------------------------------------------------------------
# WHY TRANSFER FUNCTIONS NOW EXIST
#
# The deviation system:
#
#     dx_dot = A_i dx + B du
#     dy     = C dx + D du
#
# satisfies all assumptions required for transfer functions:
#   - linear
#   - time-invariant (within a bin)
#   - zero equilibrium
#   - no constant forcing
#
# Therefore, transfer functions computed from (A_i, B, C, D) describe:
#
#     "How small input changes around this angle bin affect the outputs"
#
# This is EXACTLY what transfer functions are meant to represent.
#
# After this point, c_i should NOT be used for TF computation — it has already
# been absorbed into the equilibrium (x0_i, u0_i). Keeping it would double-count
# gravity/trim effects and produce incorrect dynamics.
#
# TL;DR:
#   Transfer functions describe local input-output dynamics about an equilibrium.
#   Constant terms set the equilibrium; they are not part of the dynamics.
# -----------------------------------------------------------------------------


def make_deviation_model_from_block(block, x0=None):
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
    A, B, c = unpack_block(block)

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
            A, B, x0, u0, _ = make_deviation_model_from_block(block)
            dev_bank[ip, it] = (x0, u0)
            bank_a[ip, it] = A
            bank_b[ip, it] = B

    return bank_a, bank_b, dev_bank


import numpy as np

def compute_Kbank(bank_a, bank_b, Q, R):
    Nphi, Ntheta = bank_a.shape[0], bank_a.shape[1]
    n = bank_a.shape[2]          # 12
    m = bank_b.shape[3]          # 4  (because bank_b is (Nphi,Ntheta,n,m))

    Kbank = np.zeros((Nphi, Ntheta, m, n))  # (Nphi,Ntheta,4,12)

    for ip in range(Nphi):
        for it in range(Ntheta):
            Ad = bank_a[ip, it]   # (n,n)
            Bd = bank_b[ip, it]   # (n,m)
            K, S, E = lqr(Ad, Bd, Q, R)    # K is (m,n)
            Kbank[ip, it, :, :] = K

    return Kbank

