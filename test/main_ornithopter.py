import numpy as np
from scipy.integrate import odeint
import time
import matplotlib.pyplot as plt

from fym.core import BaseEnv
from fym.core import Clock
from fym.models.ornithopter import Ornithopter


class TestEnv(BaseEnv):
    def __init__(self, dt=0.01, max_t=10, logging_off=True):
        systems_dict={
            "bird": Ornithopter()
            }
        self.dt = dt
        self.max_t = max_t
        self.t = 0
        super().__init__(systems_dict=systems_dict, dt=dt, max_t=max_t, logging_off=logging_off)
  
    def reset(self):
        super().reset()
        return self.observe_flat()

    def step(self, action):
        self.update(action)
        self.t += self.dt
        done = self.t > self.max_t
        return self.observe_flat(), 0, done, {}
    
    def derivs(self, time, action):
        bird = self.systems_dict['bird']
        bird.set_dot(bird.deriv(time, action))


env = TestEnv()
time_step = 0.01
t0 = 0
tf = 1
time_series = np.arange(t0, tf, time_step)

obs = env.reset()
obs_series = np.array([], dtype=np.float64).reshape(0, len(obs))
start_time = time.time() # elpased time

for i in range(np.int(tf/time_step)):
    # action = [wing_freq, wing_mag, wing_bias, delta_e, delta_r]
    action = np.array([10, 0.56, -0.13, np.deg2rad(-5), 0])
    next_obs, _, done, _ = env.step(action)

    if done:
        break
    obs = next_obs
    obs_series = np.vstack((obs_series, obs))

elapsed_time = time.time() - start_time
print('simulation time = ', tf - t0, '[s]')
print('elpased time = ', elapsed_time, '[s]')

plt.figure(0)
ylabel_list = ['x', 'y', 'z', 'u', 'v', 'w', 'phi', 'theta', 'eta', 'p', 'q', 'r']
for i in range(12):
    plt.subplot(4, 3, i + 1)
    plt.plot(time_series, obs_series[:, i])
    plt.xlabel('time [s]')
    plt.ylabel(ylabel_list[i])
plt.show()

'''
plt.figure(2)
for i in range(3):
    plt.plot(t_his, y_his[:, i])
plt.legend(('x', 'y', 'z'))
plt.figure(3)
for i in range(3):
    plt.plot(t_his, y_his[:, 6 + i])
plt.legend(('phi', 'theta', 'psi'))
plt.show()
'''