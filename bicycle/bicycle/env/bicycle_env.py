import gym
import gymnasium
import numpy as np
import matplotlib.pyplot as plt
import os
import time

class BicycleEnv(gymnasium.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}
    def __init__(self):
        # Define action and observation space
        # d action first, then T
        self.action_space = gymnasium.spaces.Box(low=np.array([-0.01, -1.0]), high=np.array([0.01, 1.0]), dtype=np.float32)
        # self.observation_space = gymnasium.spaces.Box(low=np.array([-np.pi/2,-100,-np.pi/15,-100,-np.inf, -np.inf]), high=np.array([np.pi/2,100,np.pi/15,100,np.inf, np.inf]), shape=(6,), dtype=np.float32)
        self.observation_space = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float32)
        self.reward_range = gymnasium.spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        self.state = [0, 0, 0, 0, 0, 0]

        # Other parameters needed for the environment
        self.dt = 0.01  # timestep
        # Other constants/variables
        self.c = 0.66  # Horizontal distance between the point, where the front wheel touches the ground and the CM.
        self.d_CM = 0.3  # The vertical distance between the CM for the bicycle and for the cyclist
        self.h = 0.94  # Height of the CM over the ground
        self.l = 1.11  # Distance between the front tyre and the back tyre at the point where they touch the ground
        self.Mc = 15  # Bicycle Mass
        self.Md = 1.7  # Tyre Mass
        self.Mp = 60  # Cyclist mass
        self.M = self.Mc + 2 * self.Md + self.Mp  # Total mass
        self.r = 0.34  # Tyre radius
        self.v = 10 * 3.6  # Velocity bicycle
        self.g = 9.81  # Gravity
        self.I_bc = 13 * self.Mc * self.h ** 2 / 3 + self.Mp * (self.h + self.d_CM)**2  # Moment of inertia of bicycle and cyclist
        self.I_dc = self.Md * self.r ** 2  # Various MOI of tyre
        self.I_dv = 3 * self.Md * self.r ** 2  # Various MOI of tyre
        self.I_dl = 0.5 * self.Md * self.r ** 2  # Various MOI of tyre
        self.sigma_dot = self.v / self.r  # Angular velocity of tyre

        # Initialize variables for visualization
        self.figure_save_dir = "figures/"
        self.xb_runs, self.yb_runs, self.theta_runs, self.omega_runs, self.reward_runs = [], [], [], [], []
        self.xb_history = []
        self.yb_history = []
        self.theta_history = []
        self.omega_history = []
        self.step_history = []
        self.reward_history = []
        self.current_step = 0


    def _get_info(self):
        return {
            "state": self.state
        }


    def bicycle_test(self, x, action_d, action_T, dt, l, c, h, I_bc, M, g, I_dc, sigma_dot, v, Md, r, I_dv, I_dl):

        # ------------INPUTS------------
        # d: agents choice of displacement of the CM perp. to the bicycle [scalar]
        # T: torque the agents exerts onto the handlebars [scalar]
        # x: state vector containing [theta, theta_dot, omega, omega_dot, xb, yb
        # dt: timestep of iteration

        # ------------Further defining values------------
        tan_x0 = np.tan(x[0])
        sin_x0 = np.sin(x[0])

        if np.abs(tan_x0) < 1e-10:  # Small threshold to handle nearly zero tan(x[0])
            # Handle case where tan(x[0]) is almost zero
            r_CM = l
            rb = np.inf
        else:
            r_CM = ((l - c) ** 2 + l ** 2 / tan_x0 ** 2) ** (1 / 2)
            rb = l / np.abs(np.tan(x[0]))

        if np.abs(sin_x0) < 1e-10:  # Small threshold to handle nearly zero sin(x[0])
            # Handle case where sin(x[0]) is almost zero
            rf = np.inf
        else:
            rf = l / np.abs(sin_x0)

        phi = x[2] + np.arctan(action_d / h)

        # ------------Equations of Motion------------
        omega_ddot = 1 / I_bc * (M * h * g * np.sin(phi) - np.cos(phi) * (
                    I_dc * sigma_dot * x[1] + np.sign(x[0]) * v * (Md * r / rf + Md * r / rb + M * h / r_CM)))
        omega_dot = x[3] + omega_ddot * dt
        omega = x[2] + omega_dot * dt
        theta_ddot = (action_T - I_dv * sigma_dot * omega_dot) / I_dl
        # theta_ddot = (action_T - I_dv * sigma_dot * x[3]) / I_dl
        theta_dot = x[1] + theta_ddot * dt
        theta = x[0] + theta_dot * dt

        # ------------x and y locations------------
        sine_argument = v * dt / (2 * rb)
        if rb == 0:
            # Handle case where rb is zero
            xb = x[4] - v * dt * np.sin(phi)
            yb = x[5] + v * dt * np.cos(phi)
        elif np.abs(sine_argument) <= 1.0:
            # Handle case where arcsin argument is within [-1, 1]
            xb = x[4] + v * dt * (-np.sin(phi + np.sign(phi) * np.arcsin(sine_argument)))
            yb = x[5] + v * dt * (np.cos(phi + np.sign(phi) * np.arcsin(sine_argument)))
        else:
            # Handle case where arcsin argument is outside [-1, 1]
            xb = x[4] - v * dt * np.sin(phi)
            yb = x[5] + v * dt * np.cos(phi)

        # ------------Output------------
        return np.array([theta, theta_dot, omega, omega_dot, xb, yb])


    def step(self, action):
        done = False
        # Reformat actions
        action_d, action_T = action  # TODO: fix this once the program is up and running
        # Take a step in the dynamics/ update state
        next_state = self.bicycle_test(self.state, action_d, action_T, self.dt, self.l, self.c, self.h, self.I_bc,
                                  self.M, self.g, self.I_dc, self.sigma_dot, self.v, self.Md, self.r, self.I_dv,
                                  self.I_dl)
        # Compute rewards
        if abs(next_state[2]) >= np.pi / 12:
            self.reward =-1
        else:
            self.reward = 1
        # if abs(next_state[2]) >= np.pi / 12 or abs(next_state[3]) >= 10:
        #     self.reward =-10
        #     # print('Bicycle has fallen!')
        # elif abs(next_state[2]) >= np.pi / 24 or abs(next_state[3]) >= 5:
        #     self.reward = 0
        # elif abs(next_state[3]) <= 0.25 or abs(next_state[2]) <= 0.06:
        #     self.reward = 10
        # else:
        #     self.reward = 1
        # Append for visualisation
        self.current_step += 1
        self.xb_history.append(next_state[4])
        self.yb_history.append(next_state[5])
        self.theta_history.append(next_state[0])
        self.omega_history.append(next_state[2])
        self.reward_history.append(self.reward)
        # Set termination conditions
        if self.reward <= -1:
            done = True
            self.xb_runs.append(self.xb_history)
            self.yb_runs.append(self.yb_history)
            self.theta_runs.append(self.theta_history)
            self.omega_runs.append(self.omega_history)
            self.step_history.append(self.current_step)
            self.reward_runs.append(self.reward_history)
        # return parameters
        self.state = next_state
        info = self._get_info()
        return self.state, self.reward, done, False, info


    def reset(self, seed=None):

        self.state = [0, 0, 0, 0, 0, 0]
        info = self._get_info()
        self.current_step = 0
        self.xb_history = []
        self.yb_history = []
        self.theta_history = []
        self.omega_history = []

        # Initialize the state vector here
        return np.array(self.state), {}


    def render(self):
        if not os.path.exists('figures'):
            os.makedirs('figures')

        print('Plotting figures')
        plt.figure()
        for run_num in range(len(self.xb_runs)):
            plt.plot(self.xb_runs[run_num], self.yb_runs[run_num])
        # plt.plot(self.xb_history, self.yb_history)
        plt.xlabel("xb")
        plt.ylabel("yb")
        # plt.title("xb vs yb")
        # plt.show()
        # Save the figure as an image file
        fig_filename = self.figure_save_dir + "xb_vs_yb.png"
        plt.savefig(fig_filename)
        plt.close()  # Close the figure

        # Plot step history
        plt.figure()
        plt.plot(self.step_history)
        plt.ylabel("Time upright [s]")
        # plt.title("Episode")
        # plt.show()
        # Save the figure as an image file
        fig_filename = self.figure_save_dir + "step_history.png"
        plt.savefig(fig_filename)
        plt.close()  # Close the figure

        # # Plot reward history
        # plt.figure()
        # plt.plot(self.step_history)
        # plt.ylabel("Reward")
        # plt.title("Reward History")
        # # plt.show()
        # # Save the figure as an image file
        # fig_filename = self.figure_save_dir + "reward_history.png"
        # plt.savefig(fig_filename)
        # plt.close()  # Close the figure

        # Plot theta vs time
        plt.figure()
        for run_num in range(len(self.xb_runs)):
            plt.plot(self.theta_runs[run_num], 'b')
        # plt.plot(self.time_steps, self.theta_history, label="theta")
        plt.xlabel("Time")
        plt.ylabel("Value")
        # plt.legend()
        plt.title("Theta vs Time")
        # plt.show()
        # Save the figure as an image file
        fig_filename = self.figure_save_dir + "theta_vs_time.png"
        plt.savefig(fig_filename)
        plt.close()  # Close the figure

        # Plot omega vs time
        plt.figure()
        for run_num in range(len(self.xb_runs)):
            plt.plot(self.omega_runs[run_num], 'r')
        # plt.plot(self.time_steps, self.omega_history, label="omega")
        plt.xlabel("Time")
        plt.ylabel("Value")
        # plt.legend()
        plt.title("Omega vs Time")
        # plt.show()
        # Save the figure as an image file
        fig_filename = self.figure_save_dir + "omega_vs_time.png"
        plt.savefig(fig_filename)
        plt.close()  # Close the figure

        print('Average run duration', np.mean(self.step_history))


    def close(self):
        pass
