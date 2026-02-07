from control import lqr,dlqr
import numpy as np
import matplotlib.pyplot as plt

class LQRController:
    def __init__(self,Q_bank = None,R_bank = None,A_bank = None,B_bank = None):
        self.K_bank = None
        self.K_bank_descrete = None
        self.Q_bank = Q_bank
        self.R_bank = R_bank
        self.A_bank = A_bank
        self.B_bank = B_bank
    def load_banks(self, Q_bank = None,R_bank = None,A_bank = None,B_bank = None):
        self.Q_bank = Q_bank
        self.R_bank = R_bank
        self.A_bank = A_bank
        self.B_bank = B_bank
    def gen_K_bank_continoius(self):
        # TODO, Find size of A,B

        # TODO Unpack banks

        # TODO Generate K matrixes
        K , S , E = lqr()
        
        # TODO Bank K matrixes

        return self.K_bank
    def gen_K_bank_descrete(self):
        # TODO, Find size of A,B

        # TODO Unpack banks

        # TODO Descretize Banks

        # TODO Generate K matrixes
        K , S , E = dlqr()
        
        # TODO Bank K matrixes

        return self.K_bank
    
    