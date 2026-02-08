from control import lqr,dlqr
import numpy as np
import matplotlib.pyplot as plt
from Utils import build_tan_pwl, make_deviation_model, bin_index

class LQRController:
    def __init__(self,Q,R,A_bank = None,B_bank = None,state_space = None):
        self.state_space = state_space
        self.K_bank = None
        self.K_bank_descrete = None
        self.Q = Q
        self.R = R
        self.A_bank = A_bank
        self.B_bank = B_bank
    def load_banks(self, Q = None,R = None,A_bank = None,B_bank = None):
        self.Q = Q
        self.R = R
        self.A_bank = A_bank
        self.B_bank = B_bank
    def gen_K_bank_continous(self):

        if self.Q is None or self.R is None or self.A_bank is None or self.B_bank is None:
            raise ValueError("Q, R, A_bank, and B_bank must be loaded before generating K_bank.")
        for ip in range(self.A_bank.shape[0]):
            for it in range(self.A_bank.shape[1]):
                
                A = self.A_bank[ip, it]
                B = self.B_bank[ip, it]
                K, S, E = lqr(A, B, self.Q, self.R)
                if self.K_bank is None:
                    self.K_bank = np.zeros((self.A_bank.shape[0], self.A_bank.shape[1], K.shape[0], K.shape[1]))
                self.K_bank[ip, it] = K
        return self.K_bank


    def gen_K_bank_discrete(self):
        # TODO, Find size of A,B

        # TODO Unpack banks

        # TODO Descretize Banks

        # TODO Generate K matrixes
        K , S , E = dlqr()
        
        # TODO Bank K matrixes

        return self.K_bank
    

    def lqr_controller_from_devbank(self,x):
        TORQUE_LIM = 50.0
        # pick model cell based on *current* roll/pitch (or whatever you’re indexing)
        ip = bin_index(x[6], self.state_space.edges_phi)
        it = bin_index(x[7], self.state_space.edges_th)

        x0, u0 = self.state_space.x0_bank[ip, it], self.state_space.u0_bank[ip, it]      # equilibrium for that cell
        K = self.K_bank[ip, it]              # 4x12

        dx = x - x0
        du = -K @ dx
        u = u0 + du


        return np.clip(u, -TORQUE_LIM, TORQUE_LIM)
        



    
    
class State_space:
    def __init__(self,Nphi=30, Ntheta=30,phi_max=np.deg2rad(35), theta_max=np.deg2rad(35),g=9.81, m=2.0, Ix=0.2, Iy=0.2, Iz=0.2,b_motion=-0.1,b_spinning=-0.1,states=12, 
                 inputs=4, outputs=6):
        self.Nphi = Nphi
        self.Ntheta = Ntheta    
        self.phi_max = phi_max
        self.theta_max = theta_max
        self.g = g
        self.m = m
        self.Ix = Ix
        self.Iy = Iy
        self.Iz = Iz
        self.b_motion = b_motion
        self.b_spinning = b_spinning
        self.abank = None
        self.bbank = None
        self.a_non_dev_bank = None
        self.b_non_dev_bank = None
        self.cbank = None
        self.x0_bank = None 
        self.u0_bank = None
        self.block = None
        self.edges_phi = None
        self.edges_th = None    
        self.a_phi = None
        self.b_phi = None
        self.a_th = None
        self.b_th = None
        self.states = states
        self.inputs = inputs
        self.n_outputs = outputs
        self.C = np.zeros((self.n_outputs, self.states))
        self.D = np.zeros((self.n_outputs, self.inputs))
    def gen_banks(self):
            
        # Separate tables for phi and theta
        edges_phi, a_phi, b_phi = build_tan_pwl(self.phi_max, self.Nphi)
        edges_th,  a_th,  b_th  = build_tan_pwl(self.theta_max, self.Ntheta)


        self.abank = np.zeros((self.Nphi, self.Ntheta, self.states, self.states))
        self.bbank = np.zeros((self.Nphi, self.Ntheta, self.states, self.inputs))
        self.a_non_dev_bank = np.zeros((self.Nphi, self.Ntheta, self.states, self.states))
        self.b_non_dev_bank = np.zeros((self.Nphi, self.Ntheta, self.states, self.inputs))
        self.cbank = np.zeros((self.Nphi, self.Ntheta, self.states))
        self.x0_bank = np.zeros((self.Nphi, self.Ntheta, self.states))
        self.u0_bank = np.zeros((self.Nphi, self.Ntheta, self.inputs))
        self.edges_phi = edges_phi
        self.edges_th = edges_th
        self.a_phi = a_phi
        self.b_phi = b_phi
        self.a_th = a_th
        self.b_th = b_th


       

        # velocities
        self.C[0, 3] = 1.0   # vx = xdot
        self.C[1, 4] = 1.0   # vy = ydot
        self.C[2, 5] = 1.0   # vz = zdot

        # angles
        self.C[3, 6] = 1.0   # phi
        self.C[4, 7] = 1.0   # theta
        self.C[5, 8] = 1.0   # psi

        # No direct feedthrough
        self.D = np.zeros((self.n_outputs, self.inputs))

        for ip in range(self.Nphi):
            for it in range(self.Ntheta):
                A = np.zeros((self.states, self.states))
                B = np.zeros((self.states, self.inputs))
                c = np.zeros((self.states,))

                # Kinematics
                A[0,3]=1; A[1,4]=1; A[2,5]=1
                A[6,9]=1; A[7,10]=1; A[8,11]=1

                A[3,3] = self.b_motion; A[4,4]=self.b_motion; A[5,5]=self.b_motion #wind
               
                A[9,9]=self.b_spinning; A[10,10]=self.b_spinning; A[11,11]=self.b_spinning #spining drag
                # Inputs (choose sign convention you want)
                B[5,0]  =  1.0/self.m
                #B[4,0]  =  (np.sin((edges_phi[ip]))/m)
                #B[3,0]  =  -(np.sin((edges_th[it]))/m)
                B[9,1]  =  1.0/self.Ix
                B[10,2] =  1.0/self.Iy
                B[11,3] =  1.0/self.Iz

                # Scheduled PWL sin
                # xddot = -g*tan(theta)
                A[3,7] = -self.g * a_th[it]
                c[3]   = -self.g * b_th[it] 

                # yddot =  g*tan(phi)
                A[4,6] =  self.g * a_phi[ip]
                c[4]   =  self.g * b_phi[ip]
                self.a_non_dev_bank[ip, it] = A
                self.b_non_dev_bank[ip, it] = B

                # Compute trim point for this block (x0,u0) such that A x0 + B u0 + c = 0       
                A,B,x0, u0, _ = make_deviation_model(A, B, c)
                self.x0_bank[ip, it] = x0
                self.u0_bank[ip, it] = u0

                # Pack into one block
                self.abank[ip, it] = A
                self.bbank[ip, it] = B   
                self.cbank[ip, it] = c


class Virtual_input_to_motor_inputs:
    def __init__(self,virtual_u,number_of_inputs = 4, arm_len= 0.3, yaw_c =0.02, f_min=-25, f_max=25,is_x_config=True, desat_yaw=True, iters=8):
        self.arm_len = arm_len
        self.yaw_c = yaw_c
        self.f_min = f_min
        self.f_max = f_max
        self.is_x_config = is_x_config
        self.desat_yaw = desat_yaw
        self.iters = iters
        self.virtual_u = virtual_u 
        self.u = np.zeros(number_of_inputs)
        self.f = np.zeros(number_of_inputs)
        self.M = None
        self.Minv = None
        
    def mixing_matrix_gen(self):
        # Effective arm for torque depending on geometry
        l_eff = self.arm_len / np.sqrt(2.0) if self.is_x_config else self.arm_len
        c = self.yaw_c

        # Mixer: u = M f
        # Row 1: T
        # Row 2: tau_x (roll): left(+), right(-) => [+ - + -]
        # Row 3: tau_y (pitch): front(-), rear(+) => [- - + +]
        # Row 4: tau_z (yaw): (1,4) CCW negative, (2,3) CW positive => [- + + -]
        M = np.array([
            [1.0,    1.0,    1.0,    1.0],
            [l_eff, -l_eff,  l_eff, -l_eff],
            [-l_eff,-l_eff,  l_eff,  l_eff],
            [-c,     c,      c,     -c],
        ], dtype=float)
        self.M = M
        

        # Precompute inverse 
        Minv = np.linalg.inv(M)
        self.Minv = Minv
        return Minv, M
    def mix_to_motors(self,u, Minv,destroy=False,iters=4):
        self.virtual_u = u
        self.f = Minv @ self.virtual_u
        if destroy:
            for _ in range(max(1, iters)):
                f = Minv @ self.virtual_u
                if np.all(f >= self.f_min) and np.all(f <= self.f_max):
                    break

                dz = Minv[:, 3]  # sensitivity of motor thrusts to tau_z
                over = np.maximum(f - self.f_max, 0.0)
                under = np.maximum(self.f_min - f, 0.0)

                # Choose worst violation to fix
                k_over = int(np.argmax(over))
                k_under = int(np.argmax(under))

                if over[k_over] > under[k_under]:
                    k = k_over
                    if abs(dz[k]) > 1e-12:
                        self.virtual_u[3] -= over[k] / dz[k]
                else:
                    k = k_under
                    if abs(dz[k]) > 1e-12:
                        self.virtual_u[3] += under[k] / dz[k]

        self.f = np.clip(self.f, self.f_min, self.f_max)
        self.u = self.M @ self.f
        return self.u,self.f

class Simulator:
    def __init__(self,LQR_controller = None,State_space = None,virtual_input_to_motor_inputs = None):
        self.LQR_controller = LQR_controller
        self.State_space = State_space
        self.virtual_input_to_motor_inputs = virtual_input_to_motor_inputs
        self.bank_a = self.State_space.abank
        self.bank_b = self.State_space.bbank
        self.x0_bank = self.State_space.x0_bank
        self.u0_bank = self.State_space.u0_bank
        self.K_bank = self.LQR_controller.K_bank
        self.edges_phi = self.State_space.edges_phi
        self.edges_th = self.State_space.edges_th
        self.current_state = np.zeros(self.State_space.states)
        self.next_state = np.zeros(self.State_space.states)
        self.y = np.zeros(self.State_space.n_outputs)
        self.x_hist = None
        self.y_hist = None
        self.u_hist = None
        self.f_hist = None
        self.current_deviation = None
        
    def step_state(self, u, dt):
        # 1) pick bins from current state
        ip = bin_index(self.current_state[6], self.edges_phi)   # phi
        it = bin_index(self.current_state[7], self.edges_th)    # theta

        # 2) get deviation model + trim
        A = self.bank_a[ip, it]
        B = self.bank_b[ip, it]
        self.current_deviation, u0 = self.x0_bank[ip, it], self.u0_bank[ip, it]

        # 3) deviation coordinates
        dx = self.current_state - self.current_deviation
        du = u - u0

        # 4) state derivative (deviation)
        dx_dot = A @ dx + B @ du

        # 5) integrate
        dx_next = dx + dx_dot * dt

        # 6) back to absolute state
        x_next = dx_next + self.current_deviation
        self.current_state = x_next

        return x_next
    

    def simulator(self,dt,deviation_output):
        # step
        self.x_next = self.step_state(self.virtual_input_to_motor_inputs.u, dt=dt)
        # output in deviation coordinates: y = C*dx + D*du
        ip = bin_index(self.x_next[6], self.edges_phi)
        it = bin_index(self.x_next[7], self.edges_th)
        x0, u0 = self.x0_bank[ip, it], self.u0_bank[ip, it]
        if deviation_output:

            dx = self.x_next - x0
            du = self.virtual_input_to_motor_inputs.u - u0
            y = self.State_space.C @ dx + self.State_space.D @ du
        else:
            # physical outputs directly from state
            y = self.State_space.C @ self.x_next + self.State_space.D @ self.virtual_input_to_motor_inputs.u

        return self.x_next, y,x0, u0
    
    def run_and_plot(self,T=5.0, dt=0.01, deviation_output=False):
        """
        x0: initial 12-state vector
        u_fun: function u_fun(t, x) -> (4,) input vector
        """
        self.x_next, self.y, x0, u0 = self.simulator(dt=dt, deviation_output=deviation_output)

        N = int(np.floor(T / dt)) + 1
        t = np.linspace(0.0, T, N)

        self.x_hist = np.zeros((N, self.State_space.states))
        self.y_hist = np.zeros((N, self.State_space.n_outputs))
        self.u_hist = np.zeros((N, self.State_space.inputs))
        self.f_hist = np.zeros((N, self.State_space.inputs))
        x = np.array(x0, dtype=float)
        thrust_disturbance = np.random.uniform(0,10)
        tau_z_disturbance= np.random.uniform(0,0.5)
        tau_y_disturbance= np.random.uniform(0,0.5)
        tau_x_disturbance= np.random.uniform(0,0.5)
        distrubance = np.array([thrust_disturbance, tau_x_disturbance, tau_y_disturbance, tau_z_disturbance])
        
        for k in range(N):
            if t[k] < 1.0:
                u = distrubance
                _, f = self.virtual_input_to_motor_inputs.mix_to_motors(u, self.virtual_input_to_motor_inputs.Minv, destroy=True, iters=8)
            else:
                u = self.LQR_controller.lqr_controller_from_devbank(x)
                u,f = self.virtual_input_to_motor_inputs.mix_to_motors(u, self.virtual_input_to_motor_inputs.Minv, destroy=True, iters=8)
            self.x_hist[k] = x
            self.f_hist[k] = f
            self.u_hist[k] = u

            x, y, x0, u0 = self.simulator(dt=dt, deviation_output=deviation_output)
            self.y_hist[k] = y

        labels = ["vx", "vy", "vz", "phi", "theta", "psi"]

        plt.figure()
        for i in range(6):
            plt.plot(t, self.y_hist[:, i], label=labels[i])
        plt.xlabel("time (s)")
        plt.ylabel("outputs")
        plt.legend()
        plt.grid(True)
        plt.show()

        # optional: also plot inputs
        plt.figure()
        for i, name in enumerate(["u1(net z)", "tau_x", "tau_y", "tau_z"]):
            plt.plot(t, self.u_hist[:, i], label=name)
        plt.xlabel("time (s)")
        plt.ylabel("inputs")
        plt.legend()
        plt.grid(True)
        plt.show()
        for i, name in enumerate(["f1", "f2", "f3", "f4"]):
            plt.plot(t, self.f_hist[:, i], label=name)
        plt.xlabel("time (s)")
        plt.ylabel("inputs")
        plt.legend()
        plt.grid(True)
        plt.show()

        return t, self.x_hist, self.y_hist, self.u_hist

      

if __name__ == "__main__":
    # Example usage
    Q = np.diag([
    0.0, 0.0, 0.0,     # position (ignore)
    10.0, 10.0, 10.0,# vx, vy, vz (CARE A LOT)
    5.0, 5.0, 0.2,     # roll, pitch, yaw (small, not zero)
    5.0, 5.0, 1.0      # p, q, r rates (small-moderate damping)
    ])
    
    R = np.diag([1.0, 0.2, 0.2, 0.3])

    Virtual_input_to_motor_inputs = Virtual_input_to_motor_inputs(virtual_u=np.zeros(4))
    Virtual_input_to_motor_inputs.mixing_matrix_gen()
    State_space = State_space()
    State_space.gen_banks()
    lqr_controller = LQRController(Q,R,state_space=State_space)
    lqr_controller.load_banks(Q,R,State_space.abank,State_space.bbank)
    K_bank = lqr_controller.gen_K_bank_continous()
    Simulator = Simulator(LQR_controller=lqr_controller,State_space=State_space,virtual_input_to_motor_inputs=Virtual_input_to_motor_inputs)    
    Simulator.run_and_plot()

    print("K_bank shape:", K_bank.shape)