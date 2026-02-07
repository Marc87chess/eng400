from control import lqr,dlqr
import numpy as np
import matplotlib.pyplot as plt
from Utils import build_tan_pwl, make_deviation_model

class LQRController:
    def __init__(self,Q,R,A_bank = None,B_bank = None):
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
    



    
    
class State_space:
    def __init__(self,Nphi=30, Ntheta=30,phi_max=np.deg2rad(35), theta_max=np.deg2rad(35),g=9.81, m=2.0, Ix=0.2, Iy=0.2, Iz=0.2,b_motion=-0.1,b_spinning=-0.1,states=12, 
                 inputs=4):
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



class Simulator:
    def __init__(self):
        pass
    
