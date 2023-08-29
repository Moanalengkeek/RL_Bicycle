import numpy as np
import json

c = 0.66  # Horizontal distance between the point, where the front wheel touches the ground and the CM.
d_CM = 0.3  # The vertical distance between the CM for the bicycle and for the cyclist
h = 0.94  # Height of the CM over the ground
l = 1.11  # Distance between the front tyre and the back tyre at the point where they touch the ground
Mc = 15  # Bicycle Mass
Md = 1.7  # Tyre Mass
Mp = 60  # Cyclist mass
M = Mc+2*Md+Mp  # Total mass
r = 0.34  # Tyre radius
v = 15 / 3.6  # Velocity bicycle
g = 9.81  # Gravity
I_bc = 13 * Mc * h ** 2 / 3 + Mp * (h + d_CM)**2  # Moment of inertia of bicycle and cyclist
I_dc = Md*r**2  # Various MOI of tyre
I_dv = 3 * Md*r**2  # Various MOI of tyre
I_dl = 0.5*Md*r**2  # Various MOI of tyre
sigma_dot = v/r  # Angular velocity of tyre

T_options = [-2, 0,  2]  # Torque applied to the handlebars
d_options = [-0.02, 0, 0.02]  # Displacement of the center of mass of the cyclist
noise = 0.02*np.random.uniform(-1,1)  # Will be added to the displacement to encourage exploration/simulate skill
actions = len(T_options)*len(d_options)  # Number of actions available to the agent
states = 7*7*7*7  # Number of states available

epsilon = 0.9  # Determines how much is exploration and how much is exploitation

# assume the front fork remains vertical

# theta = angle of the bike direction from froward
# omega = angle the bicycle is tilted from vertical
# phi = total angle of tilt of the center of mass

def bicycle(x,action_d,action_T,dt):
    # Function that models the behaviour of the bicycle at a point in time.
    T = T_options[int(action_T)]  # Transform from integer representation to value
    d = d_options[int(action_d)]  # Transform from integer representation to value

    # ------------INPUTS------------
    # d: agents choice of displacement of the CM perp. to the bicycle [scalar]
    # T: torque the agents exerts onto the handlebars [scalar]
    # x: state vector containing [theta, theta_dot, omega, omega_dot, loc]
    #      where loc is tyre locations: [xf, yf, xb, yb]
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

    phi = x[2] + np.arctan(d / h)

    # r_CM = ((l-c)**2+l**2/(np.tan(x[0]))**2)**(1/2)
    # rf = l/np.abs(np.sin(x[0]))
    # rb = l/np.abs(np.tan(x[0]))
    # phi = x[2] + np.arctan(d/h)


    # ------------Equations of Motion------------
    omega_ddot = 1/I_bc * (M*h*g* np.sin(phi)-np.cos(phi)*(I_dc * sigma_dot * x[1] + np.sign(x[0])*v*(Md*r/rf+Md*r/rb+M*h/r_CM)))
    omega_dot = x[3] + omega_ddot * dt
    omega = x[2] + omega_dot * dt
    # theta_ddot = (T-I_dv*sigma_dot)/I_dl
    theta_ddot = (T - I_dv * sigma_dot * omega_dot) / I_dl
    # theta_ddot = (T - I_dv * sigma_dot * x[3]) / I_dl
    theta_dot = x[1] + theta_ddot*dt
    theta = x[0] + theta_dot*dt

    # ------------x and y locations------------
    xf = x[4][0] + v * dt * (-np.sin(phi+theta+np.sign(phi+theta)*np.arcsin(v*dt/(2*rf))))
    yf = x[4][1] + v * dt * (np.cos(phi+theta+np.sign(phi+theta)*np.arcsin(v*dt/(2*rf))))
    xb = x[4][2] + v * dt * (-np.sin(phi+np.sign(phi)*np.arcsin(v*dt/(2*rb))))
    yb = x[4][3] + v * dt * (np.cos(phi+np.sign(phi)*np.arcsin(v*dt/(2*rb))))
    loc = [xf, yf, xb, yb]

    # ------------Output------------
    return [theta, theta_dot, omega, omega_dot, loc]


def discrete_state_vector():
    # Function that defines the discrete interval boxes

    # ------------Discrete interval boxes------------
    D_theta = [-np.pi / 2, -1, -0.2, 0, 0.2, 1, np.pi / 2]
    D_theta_dot = [-100, -2, -1, 0, 1, 2, 100]
    D_omega = [-np.pi / 15, -0.15, -0.06, 0, 0.06, 0.15, np.pi / 15]
    D_omega_dot = [-100, -0.5, -0.25, 0, 0.25, 0.5, 100]

    return [D_theta, D_theta_dot, D_omega, D_omega_dot]


def discretize_bicycle(x_to_discretise):
    # Function that takes the modeled bicycle behaviour and maps it to one of the discrete interval boxes

    # ------------INPUTS------------
    # x: state vector containing [theta, theta_dot, omega, omega_dot, loc]
    #      where loc is tyre locations: [xf, yf, xb, yb]
    # xd: discrete state vector

    # ------------Place into the bins------------
    xd = discrete_state_vector()
    x_discrete = [0,0,0,0,x_to_discretise[4]]
    for i in range(4):
        if x_to_discretise[i] <= xd[i][1]:
            x_discrete[i] = xd[i][0]
        elif x_to_discretise[i] <= xd[i][2]:
            x_discrete[i] = xd[i][1]
        elif x_to_discretise[i] <= xd[i][3]:
            x_discrete[i] = xd[i][2]
        elif x_to_discretise[i] >= xd[i][5]:
            x_discrete[i] = xd[i][6]
        elif x_to_discretise[i] >= xd[i][4]:
            x_discrete[i] = xd[i][5]
        elif x_to_discretise[i] >= xd[i][3]:
            x_discrete[i] = xd[i][4]
        else:
            x_discrete[i] = xd[i][3]
    # x[0] = np.digitize(x[0], xd[0])
    # x[1] = np.digitize(x[1], xd[1])
    # x[2] = np.digitize(x[2], xd[2])
    # x[3] = np.digitize(x[3], xd[3])

    # ------------Output------------
    return x_discrete


def discrete_to_continous(x):
    # Function that takes a vector mapped into bins, and returns the value based on the original mapping

    # ------------INPUTS------------
    # x: discrete state vector containing [theta, theta_dot, omega, omega_dot, loc]
    #      where loc is tyre locations: [xf, yf, xb, yb]
    #      and where theta, theta_dot, omega, and omega_dot are integers from 0-6

    # ------------Mapping Back------------
    xd = discrete_state_vector()
    x[0] = xd[0][int(x[0])]
    x[1] = xd[0][int(x[1])]
    x[2] = xd[0][int(x[2])]
    x[3] = xd[0][int(x[3])]

    # ------------Output------------
    return x

def exploration_reward(xd):
    # Function that determines the reward for a given state

    # ------------INPUTS------------
    # x: discrete state vector containing [theta, theta_dot, omega, omega_dot, loc]
    #      where loc is tyre locations: [xf, yf, xb, yb]
    #      and where theta, theta_dot, omega, and omega_dot are integers from 0-6

    # ------------Defining rewards------------
    # if xd[2] == -np.pi / 15:
    #     reward = -1
    # else:
    #     reward = 0

    # ------------Rewards on omega------------
    if xd[2] == -np.pi / 15 or xd[2] == -0.15 or xd[2] == 0.15 or xd[2] == np.pi / 15:
        # This means the cyclist has failed to stay upright
        reward = -10
        # print('Bicycle has fallen!')

    elif xd[2] == 0:
        # Bicycle is perfectly upright
        reward = 5

    elif xd[2] == -0.06 or xd[2] == 0.06:
        # Bicycle is close to being perfectly upright
        reward = 1

    else:
        reward = 0

    # ------------Rewards on omega dot and ddot------------
    if abs(xd[3]) >= 50:
        # This means the bicycle has a fast angular acceleration
        reward += -5

    elif xd[3] == 0:
        # Bicycle has no angular acceleration
        reward += 2

    # ------------Determine max reward------------
    # if reward >= 95:
    #     reward = 94
    #
    # elif reward <=-101:
    #     reward = -100
    # ------------Output------------
    return reward


def initialise_q_matrix(states, actions):
    # Function that creates the q_table depending on how many actions and states there are
    return np.zeros((states,actions))

def vector_of_states():
    # Function that creates a nx4 matrix where each row represents a unique state
    xd = discrete_state_vector()
    y = np.ones((7*7*7*7,4))
    i = 0
    for a in xd[0]:
        for b in xd[1]:
            for c in xd[2]:
                for d in xd[3]:
                    y[i, 0] = a
                    y[i, 1] = b
                    y[i, 2] = c
                    y[i, 3] = d
                    i += 1
    return y

def state_to_integer(x):
    # Function that takes the discrete state variables, and maps them to a unique position in a vector of all states

    # ------------INPUTS------------
    # x: discrete state vector containing [theta, theta_dot, omega, omega_dot, loc]
    #      where loc is tyre locations: [xf, yf, xb, yb]
    #      and where theta, theta_dot, omega, and omega_dot are integers from 0-6


    ## need to make sure the x variable is in positions, or make the y in actual state values
    # ------------Determining the index------------
    y = vector_of_states()
    for i in range(len(y)):
        if np.array_equal(x[0:4],y[i]):
            row_index = i
        # else:
        #     print(i, y[i], x[0:4])

    # ------------Output------------
    # x[4] represents loc, row_index is what we want
    return row_index, x[4]


def integer_to_state(row_index,loc):
    # Function that takes the row index of the vector of all states, and maps back to the discrete state variables

    # ------------INPUTS------------
    # row_index: integer index of row in the vector of states
    # loc: part of the state vector that needs to be kept, but not relevant here

    # ------------Reverting to the state------------
    y = vector_of_states()
    x = [1,1,1,1,1]
    x[0:4] =list(y[row_index])
    x[4] = loc

    # ------------Output------------
    return x


def vector_of_actions():
    y = np.ones((len(d_options)*len(T_options),2))
    i = 0
    for a in range(3):
        for b in range(3):
            y[i, 0] = a
            y[i, 1] = b
            i += 1
    return y


def action_to_integer(action_d, action_T):
    # Function to map each combination of actions to a unique integer

    # ------------INPUT------------
    # action_d: index of chosen d action
    # action_T: index of chosen T action

    # ------------Determining the index------------
    y = vector_of_actions()
    for i in range(len(y)):
        if np.array_equal([action_d, action_T],y[i]):
            column_index = i

    # ------------Output------------
    return column_index

def determine_choice(i,q):
    # Function used to determine which action is to be made to maximise the reward

    # ------------INPUT------------
    # i: the integer representation of the current state
    # q: the Q-table

    # ------------Determining action with the max reward------------
    action = np.argmax(q[i,:])

    # ------------Transforming back from integer------------
    y = vector_of_actions()
    numd, numT = y[int(action)]

    # ------------Output------------
    # action: the integer representation of the action (j in main code)
    # numd, numT: the indexs of the d and T actions respectively
    return action, numd, numT


def epsilon_greedy_policy(state, Q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.randint(actions)  # Choose a random action
    else:
        return np.argmax(Q_values[state_to_integer(state)])


def save_data(list1, list2, list3, list4, list5):
    List = list1.append(list2.append(list3.append(list4.append(list5))))
    with open('saved_list.json', 'w') as file:
        json.dump(List, file)
    return


def graph_data():
    with open('saved_list.json', 'r') as file:
        List = json.load(file)
    length = int(len(List)/5)
    index = length
    fig, ax = plt.subplots(2, 2)
    for run_num in range(length):
        ax[0, 0].plot(List[index:index+length][run_num])
    index += length
    for run_num in range(length):
        ax[1, 0].plot(List[index:index+length][run_num])
    index += length
    for run_num in range(length):
        ax[0, 1].plot(List[index:index+length][run_num], List[index+length:index+2*length][run_num])
    index += 2*length
    ax[1, 1].plot(List[index:index+length], 'g', label='steps')  # row=1, col=1
    plt.show()
    return




