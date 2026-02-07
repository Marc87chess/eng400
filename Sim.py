import numpy as np
from matrix_gen import *
import matplotlib.pyplot as plt
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


def step_state(x, u, dt, bank_a, bank_b, dev_bank, edges_phi, edges_th):
    # 1) pick bins from current state
    ip = bin_index(x[6], edges_phi)   # phi
    it = bin_index(x[7], edges_th)    # theta

    # 2) get deviation model + trim
    A = bank_a[ip, it]
    B = bank_b[ip, it]
    x0, u0 = dev_bank[ip, it]

    # 3) deviation coordinates
    dx = x - x0
    du = u - u0

    # 4) state derivative (deviation)
    dx_dot = A @ dx + B @ du

    # 5) integrate
    dx_next = dx + dx_dot * dt

    # 6) back to absolute state
    x_next = dx_next + x0

    return x_next

def output_state(x, C, D, u):
    y = C @ x + D @ u
    return y



def build_simulator(dt=0.01, deviation_output=False):
    global K_bank
    # Precompute banks
    Q = np.diag([
    100.0, 100.0, 100.0,     # position (ignore)
    0.0, 0.0, 0.0,# vx, vy, vz (CARE A LOT)
    2.0, 2.0, 0.2,     # roll, pitch, yaw (small, not zero)
    5.0, 5.0, 1.0      # p, q, r rates (small-moderate damping)
    ])
    
    R = np.diag([1.0, 0.2, 0.2, 0.3])
    bank, edges_phi, edges_th = precompute_bank_2d(Nphi=30, Ntheta=30)
    bank_a, bank_b, dev_bank = make_all_devation_models(bank)
    K_bank = compute_Kbank(bank_a, bank_b, Q=Q, R=R)
    def simulator(x, u):
        # step
        x_next = step_state(x, u, dt, bank_a, bank_b, dev_bank, edges_phi, edges_th)

        if deviation_output:
            # output in deviation coordinates: y = C*dx + D*du
            ip = bin_index(x_next[6], edges_phi)
            it = bin_index(x_next[7], edges_th)
            x0, u0 = dev_bank[ip, it]
            dx = x_next - x0
            du = u - u0
            y = C @ dx + D @ du
        else:
            # physical outputs directly from state
            y = C @ x_next + D @ u

        return x_next, y

    return simulator, K_bank, dev_bank, edges_phi, edges_th




def run_and_plot(x0, T=5.0, dt=0.01, deviation_output=True):
    global K_bank, distrubance
    """
    x0: initial 12-state vector
    u_fun: function u_fun(t, x) -> (4,) input vector
    """
    sim, K_bank, dev_bank, edges_phi, edges_th = build_simulator(dt=dt, deviation_output=deviation_output)

    N = int(np.floor(T / dt)) + 1
    t = np.linspace(0.0, T, N)

    x_hist = np.zeros((N, 12))
    y_hist = np.zeros((N, 6))
    u_hist = np.zeros((N, 4))

    x = np.array(x0, dtype=float)

    for k in range(N):
        if t[k] < 1.0:
            u = distrubance
        else:
            u = lqr_controller_from_devbank(x, dev_bank, K_bank, edges_phi, edges_th)
        x_hist[k] = x
        u_hist[k] = u

        x, y = sim(x, u)
        y_hist[k] = y

    labels = ["vx", "vy", "vz", "phi", "theta", "psi"]

    plt.figure()
    for i in range(6):
        plt.plot(t, y_hist[:, i], label=labels[i])
    plt.xlabel("time (s)")
    plt.ylabel("outputs")
    plt.legend()
    plt.grid(True)
    plt.show()

    # optional: also plot inputs
    plt.figure()
    for i, name in enumerate(["u1(net z)", "tau_x", "tau_y", "tau_z"]):
        plt.plot(t, u_hist[:, i], label=name)
    plt.xlabel("time (s)")
    plt.ylabel("inputs")
    plt.legend()
    plt.grid(True)
    plt.show()

    return t, x_hist, y_hist, u_hist


x0 = np.zeros(12) 
# Dead simple PID
integral = np.array([0.0, 0.0, 0.0, 0.0])
prev_error = np.array([0.0,0.0, 0.0, 0.0])

# Tuning gains
Kp = np.array([10, 10, 10, 10])
Ki = np.array([10,1, 1, 1])
Kd = np.array([1,1, 1, 0.7])

dt = 0.01  # time step
torques = None
def controller_PID(t, y, y_des):
    global integral, prev_error, torques
    
    if t < 1.0:
        return np.array([10.0, 0.01, 0.1, 0.0001])
    else:

        vx_angle = y_des[3]
        vy_angle = y_des[4]
        if torques is None:
            torques = np.array([0.0, 0.0, 0.0, 0.0])
        y_des[3] = vx_angle/(9.81)
        y_des[4] = -vy_angle/(9.81)
      
        # Error
        error = y_des[2:6] - y[2:6]  # [roll, pitch, yaw] error
        
        # PID terms
        integral += error * dt
        derivative = (error - prev_error) / dt
        prev_error = error.copy()
        
        torques = Kp * error + Ki * integral + Kd * derivative
        for i, torque in enumerate(torques):
            if abs(torque) > 50:
                if torque < 0:
                    torques[i] = -50
                else:
                    torques[i] = 50

        return np.array([torques[0], torques[1], torques[2], torques[3]])
    

thrust_disturbance = np.random.uniform(0,10)
tau_z_disturbance= np.random.uniform(0,0.5)
tau_y_disturbance= np.random.uniform(0,0.5)
tau_x_disturbance= np.random.uniform(0,0.5)
distrubance = np.array([thrust_disturbance, tau_x_disturbance, tau_y_disturbance, tau_z_disturbance])






TORQUE_LIM = 50.0

def lqr_controller_from_devbank(x, dev_bank, Kbank, edges_phi, edges_th):
    # pick model cell based on *current* roll/pitch (or whatever you’re indexing)
    ip = bin_index(x[6], edges_phi)
    it = bin_index(x[7], edges_th)

    x0, u0 = dev_bank[ip, it]      # equilibrium for that cell
    K = Kbank[ip, it]              # 4x12

    dx = x - x0
    du = -K @ dx
    u = u0 + du


    return np.clip(u, -TORQUE_LIM, TORQUE_LIM)












run_and_plot(x0, T=10.0, dt=0.01, deviation_output=False)






