import numpy as np
from numpy import sin, cos

class Ornithopter():
    def __init__(self, X0):
        self.mass = 0.45  # total mass [kg]
        self.J_x = 7.5e-3 # inertia of moment J_xx [kg*m^2]
        self.J_y = 12e-3  # inertia of moment J_yy [kg*m^2]
        self.J_z = 12e-3  # inertia of moment J_zz [kg*m^2]
        self.inertia = np.diag([self.J_x, self.J_y, self.J_z]) # inertia matrix
        self.S_w = 0.30   # wing area [m^2]
        self.b_w = 1.22   # wing span [m^2]
        self.c_w = 0.29   # wing mean aerodynamic chord [m]
        self.X = X0       # state
        self.rho = 1.2250 # air density [kg/m^3]
        self.S_t = 0.04   # tail area [m^2]
        self.b_t = 0.20   # tail span [m]
        self.c_t = 0.20   # tail mean aerodynamic chord [m]
        self.tail_coeff = {"C_x_0": -0.3181, "C_x_alp2": -0.2310,
                           "C_y_alpbet": 0.1153,
                           "C_z_0": 0.3346, "C_z_alp": -0.2729, "C_z_bet": 0.0884,
                           "C_l_bet": -0.0054, "C_l_alpbet": -0.0161,
                           "C_m_0": -0.3486, "C_m_alp": -3.3182, "C_m_bet": 0.0975, "C_m_bet2": -0.4184, "C_m_alpV": 0.3053,
                           "C_n_alpbet": -0.5094} # aerodynamic coefficients of tail
        self.wing_coeff = {"C_x_0": 0.2262, "C_x_del_w": -0.1010, "C_x_q": -0.1913, "C_x_del_w_dot": 0.0127, "C_x_del_w2": -1.4455,
                           "C_z_0": -0.9222, "C_z_q": -1.8822, "C_z_del_w_dot": -0.0627,
                           "C_m_0": 0.0955, "C_m_del_w": 0.0862} # aerodynamic coefficient of wing
    
    def deriv(self, t, U):
        """
        ---States---
        X = [x, y, z, u, v, w, phi, theta, eta, p, q, r]
        r = [x, y, z], V = [u, v, w], Phi = [phi, theta, eta]
        omega = [p, q, r]

        ---Control inputs---
        U = [wing_freq, wing_mag, wing_bias, delta_e, delta_r]

        """
        V = self.X[3:6]
        phi = self.X[6]
        theta = self.X[7]
        eta = self.X[8]
        omega = self.X[9:12]

        del_w, del_w_dot = self.flapping(t, U[0], U[1], U[2])
        U_new = np.array([del_w, U[3], U[4], del_w_dot])

        F, tau = self.force_moment(U_new)

        R_BI = rotation(1, phi)*rotation(2, theta)*rotation(3, eta)
        r_dot = R_BI.dot(V)
        v_dot = -np.cross(omega, V) + 1/self.mass*F
        E_BI = np.array([
            [cos(theta)*cos(eta), sin(eta), 0],
            [-cos(theta)*sin(eta), cos(eta), 0],
            [sin(theta), 0, 1]
        ])
        Phi_dot = np.linalg.solve(E_BI, omega)
        omega_dot = np.linalg.solve(self.inertia, -np.cross(omega, self.inertia.dot(omega)) + tau)
        X_dot = np.hstack((r_dot, v_dot, Phi_dot, omega_dot))
        return X_dot

    def flapping(self, t, freq, mag, bias):
        del_w = mag*np.sin(2*np.pi*freq*t) + bias
        del_w_dot = 2*np.pi*freq*mag*np.cos(2*np.pi*freq*t)
        return del_w, del_w_dot

    def force_moment(self, U_new):
        F = np.zeros(3)
        tau = np.zeros(3)
        V = self.X[3:6]
        Q = self.rho*np.linalg.norm(V)**2/2
        alp_T, bet_T = self.alpbet_tail(U_new)
        C_F_T, C_tau_T = self.aero_tail(alp_T, bet_T)
        C_F_W, C_tau_W = self.aero_wing(U_new)
        F = C_F_T*Q*self.S_t + C_F_W*Q*self.S_w
        tau = C_tau_T*Q*self.S_t*np.array([self.b_t, self.c_t, self.b_t]) + C_tau_W*Q*self.S_t*np.array([self.b_w, self.c_w, self.b_w])
        return F, tau

    def alpbet_tail(self, U_new):
        del_e = U_new[1]
        del_r = U_new[2]
        V = self.X[3:6]
        R_x = np.array([[1, 0, 0], [0, np.cos(del_r), np.sin(del_r)], [0, -np.sin(del_r), np.cos(del_r)]])
        R_y = np.array([[np.cos(del_e), 0, -np.sin(del_e)], [0, 1, 0], [np.sin(del_e), 0, np.cos(del_e)]])
        V_T = np.matmul(R_x, R_y).dot(V)
        alp_T = np.arctan2(V_T[2], V_T[0])
        bet_T = np.arcsin(V_T[1]/np.linalg.norm(V_T))
        return alp_T, bet_T


    def aero_tail(self, alp_T, bet_T):
        V = np.linalg.norm(self.X[3:6])
        C_x = self.tail_coeff['C_x_0'] + self.tail_coeff['C_x_alp2']*alp_T**2
        C_y = self.tail_coeff['C_y_alpbet']*alp_T*bet_T
        C_z = self.tail_coeff['C_Z_0'] + self.tail_coeff['C_Z_alp']*alp_T + self.tail_coeff['C_z_bet']*bet_T
        C_l = self.tail_coeff['C_l_bet']*bet_T + self.tail_coeff['C_l_alpbet']*alp_T*bet_T
        C_m = self.tail_coeff['C_m_0'] + self.tail_coeff['C_m_alp']*alp_T + self.tail_coeff['C_m_bet']*bet_T\
            + self.tail_coeff['C_m_bet2']*bet_T**2 + self.tail_coeff['C_m_alpV']*alp_T*V
        C_n = self.tail_coeff['C_n_bet']*bet_T
        C_F_T = np.array([C_x, C_y, C_z])
        C_tau_T = np.array([C_l, C_m, C_n])
        return C_F_T, C_tau_T


    def aero_wing(self, U):
        q = self.X[10]
        del_w = U[0]
        del_w_dot = U[3]
        C_x = self.wing_coeff['C_x_0'] + self.wing_coeff['C_x_del_w']*del_w + self.wing_coeff['C_x_q']*q\
            + self.wing_coeff['C_x_del_w_dot']*del_w_dot + self.wing_coeff['C_x_del_w2']*del_w**2
        C_z = self.wing_coeff['C_z_0'] + self.wing_coeff['C_z_q']*q + self.wing_coeff['C_z_del_w_dot']*del_w_dot
        C_m = self.wing_coeff['C_m_0'] + self.wing_coeff['C_m_del_w_dot']*del_w_dot
        C_F_W = np.array([C_x, 0, C_z])
        C_tau_W = np.array([0, C_m, 0])
        return C_F_W, C_tau_W