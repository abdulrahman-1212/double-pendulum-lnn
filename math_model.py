import torch
import numpy as np
import os
import pandas as pd  # Import pandas for saving data

# Constants
g = 9.81  # Gravity
l1 = 1.0  # Length of first pendulum
l2 = 1.0  # Length of second pendulum
m1 = 1.0  # Mass of first pendulum
m2 = 1.0  # Mass of second pendulum

# Function to compute the Lagrangian of the double pendulum
def lagrangian(state):
    q1, q2, q1_dot, q2_dot = state  # State: [theta1, theta2, omega1, omega2]

    # Kinetic energy (T)
    T1 = 0.5 * m1 * (l1 * q1_dot) ** 2
    T2 = 0.5 * m2 * ((l1 * q1_dot * np.cos(q1) + l2 * q2_dot * np.cos(q2)) ** 2 +
                     (l1 * q1_dot * np.sin(q1) + l2 * q2_dot * np.sin(q2)) ** 2)
    T = T1 + T2

    # Potential energy (V)
    V1 = m1 * g * l1 * (1 - np.cos(q1))
    V2 = m2 * g * (l1 * (1 - np.cos(q1)) + l2 * (1 - np.cos(q2)))
    V = V1 + V2

    # Lagrangian (L = T - V)
    return T - V

# Function to compute the equations of motion
def equations_of_motion(state):
    q1, q2, q1_dot, q2_dot = state
    delta = q2 - q1

    # Equations derived from the Lagrangian
    q1_ddot = (-g * (2 * m1 + m2) * np.sin(q1) - m2 * g * np.sin(q1 - 2 * q2) -
                2 * np.sin(delta) * m2 * (q2_dot**2 * l2 + q1_dot**2 * l1 * np.cos(delta))) / \
               (l1 * (2 * m1 + m2 - m2 * np.cos(2 * delta)))

    q2_ddot = (2 * np.sin(delta) * (q1_dot**2 * l1 * (m1 + m2) +
                g * (m1 + m2) * np.cos(q1) + q2_dot**2 * l2 * m2 * np.cos(delta))) / \
               (l2 * (2 * m1 + m2 - m2 * np.cos(2 * delta)))

    return np.array([q1_dot, q2_dot, q1_ddot, q2_ddot])

# Generate training data using Runge-Kutta method
def generate_data(initial_state, time_span, dt):
    num_steps = int(time_span / dt)
    data = np.zeros((num_steps, 4))
    state = initial_state.copy()

    for i in range(num_steps):
        data[i] = state
        k1 = dt * equations_of_motion(state)
        k2 = dt * equations_of_motion(state + 0.5 * k1)
        k3 = dt * equations_of_motion(state + 0.5 * k2)
        k4 = dt * equations_of_motion(state + k3)

        state += (k1 + 2 * k2 + 2 * k3 + k4) / 6  # Update state

    return data

# Generate data
initial_state = np.array([np.pi / 4, np.pi / 4, 0, 0])  # [theta1, theta2, omega1, omega2]
time_span = 10.0  # seconds
dt = 0.01  # time step
data = generate_data(initial_state, time_span, dt)

# Convert data to PyTorch tensors
input_data = torch.tensor(data[:-1], dtype=torch.float32)  # State
true_output = torch.tensor(data[1:], dtype=torch.float32)  # Next state

# Save the input data and true output locally
data_dir = './data'  # Path to data folder
os.makedirs(data_dir, exist_ok=True)  # Create directory if it doesn't exist

# Save input data and true output as CSV
np.savetxt(os.path.join(data_dir, 'input_data.csv'), input_data.numpy(), delimiter=',')
np.savetxt(os.path.join(data_dir, 'true_output.csv'), true_output.numpy(), delimiter=',')
