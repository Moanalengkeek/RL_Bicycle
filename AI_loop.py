from dynamics import *
from random import randint, uniform
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.inf)
import time

# https://medium.com/analytics-vidhya/reinforcement-learning-using-q-learning-to-drive-a-taxi-5720f7cf38df

# ------------Setting necessary values------------
q = initialise_q_matrix(states,actions)
# Read from file
# q = np.load('q_table_addition.npy')
eligibility_traces = np.zeros_like(q)
print('q_table generated!')
y = vector_of_states()
dt = 0.05

gamma = 0.99  # Discount factor
alpha = 0.5  # Learning rate
lamda = 0.95  # Eligibility trace parameter

# Create lists to store variables
theta_store = []
omega_store = []
xf_store = []
yf_store = []
reward_store = []
upright_store = []
theta_history = []
omega_history = []
xf_history = []
yf_history = []

# ------------Building the q_table------------
for a in range(100):
    # print('Reset')
    # x = [0,0,0,0,[0,0,0,0]]
    x = [0.01*uniform(-2,2), 0.001*uniform(-2,2), 0.01*uniform(-2,2), 0.001*uniform(-2,2), [0, 0, 0, 0]]
    xd = discretize_bicycle(x)
    i, loc = state_to_integer(xd)
    reward = 0
    length_upright = 0
    action_T = randint(0, 2)
    action_d = randint(0, 2)
    j = action_to_integer(action_d, action_T)
    start_time = time.time()
    if a>30:
        dt = 0.01
    while abs(x[2]) <= np.pi/16 and length_upright <= 1000:
        length_upright += 1

        theta_store.append(x[0] * 180 / np.pi)
        omega_store.append(x[2] * 180 / np.pi)
        xf_store.append(x[4][0])
        yf_store.append(x[4][1])

        # State determination
        next_x = bicycle(x, action_d, action_T, dt)
        next_xd = discretize_bicycle(next_x)
        next_i, next_loc = state_to_integer(next_xd)

        # Action determination
        epsilon_tradeoff = uniform(0,1)
        if epsilon_tradeoff >= epsilon:  # Exploitation if random value is greater than epsilon
            next_j, next_action_d, next_action_T = determine_choice(i,q)
        else:  # Otherwise exploration
            next_action_T = randint(0,2)
            next_action_d = randint(0,2)
            next_j = action_to_integer(next_action_d, next_action_T)
        reward = exploration_reward(next_xd)
        delta = reward + gamma * q[next_i,next_j] - q[i,j]
        eligibility_traces[i,j] += 1
        q[i,j] += alpha * delta * eligibility_traces[i, j]
        eligibility_traces[i, j] *= gamma * lamda

        x = next_x
        action_d = next_action_d
        action_T = next_action_T
        i = next_i
        j = next_j

    upright_store.append(length_upright)
    theta_history.append(theta_store)
    omega_history.append(omega_store)
    xf_history.append(xf_store)
    yf_history.append(yf_store)
    theta_store = []
    omega_store = []
    xf_store = []
    yf_store = []
    reward_store = []
    end_time = time.time()
    print("elapsed time = ", -start_time+end_time, " [s]")
    print(length_upright)
    # print('Bicycle has fallen')

np.save('q_table_addition  ', q) # Save to file
# ------------Plotting intermediate results------------

fig, ax = plt.subplots(2, 2)
ax[0, 1].set_xlabel("Front Tyre x-location")
ax[0, 1].set_ylabel("Front Tyre y-location")
for run_num in range(len(xf_history)):
    ax[0, 0].plot(theta_history[run_num], 'g')
for run_num in range(len(xf_history)):
    ax[1, 0].plot(omega_history[run_num], 'g')
for run_num in range(len(xf_history)):
    ax[0, 1].plot(xf_history[run_num], yf_history[run_num], 'g')
ax[1, 1].plot(upright_store, 'g', label='steps') #row=1, col=1
ax[1, 1].set_xlabel("No. of Episodes of Learning")
ax[1, 1].set_ylabel("No. of Steps per Episode")
plt.show()


save_data(theta_history, omega_history, xf_history, yf_history, upright_store)
# # ------------One run to generate plottable results------------
# epsilon = 0.1  # Set value of epsilon to allow for more exploitation
# x = [0.1 * randint(-2, 2), 0.1 * randint(-2, 2), 0.01 * randint(-2, 2), 0.01 * randint(-2, 2), [0, 0, 0, 0]]
# xd = discretize_bicycle(x)
# i, loc = state_to_integer(xd)
# reward = 0
# length_upright = 0
# while reward > -5 and length_upright <= 100:
#     length_upright += 1
#     epsilon_tradeoff = uniform(0, 1)
#     if epsilon_tradeoff >= epsilon:  # Exploitation if random value is greater than epsilon
#         j, action_d, action_T = determine_choice(i, q)
#     else:  # Otherwise exploration
#         action_T = randint(0, 2)
#         action_d = randint(0, 2)
#         j = action_to_integer(action_d, action_T)
#     x = bicycle(x, action_d, action_T, dt)
#     xd = discretize_bicycle(x)
#     i, loc = state_to_integer(xd)
#     # print(i,j)
#     reward = exploration_reward(xd, q[i, j])
#     print(reward)
#     # print()
#     q[i, j] = reward
#     theta_store.append(x[0])
#     omega_store.append(x[2])
#     xf_store.append(x[4][0])
#     yf_store.append(x[4][1])
#     # reward_store.append(reward)
#
# # ------------Plotting intermediate results------------
# fig, ax = plt.subplots(2, 2)
# ax[0, 0].plot(theta_store, 'r', label='theta') #row=0, col=0
# # ax[0, 0].plot(reward_store, 'bo')
# ax[1, 0].plot(omega_store, 'r', label='omega') #row=1, col=0
# # ax[1, 0].plot(reward_store, 'bo')
# ax[0, 1].plot(xf_store, 'g', label='xf') #row=0, col=1
# # ax[0, 1].plot(reward_store, 'bo')
# ax[1, 1].plot(yf_store, 'g', label='yf') #row=1, col=1
# # ax[1, 1].plot(reward_store, 'bo')
# plt.show()


