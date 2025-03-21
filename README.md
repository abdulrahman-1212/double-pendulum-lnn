---

# Modelling Double Pendulum with Lagrangian Neural Network

## Overview

This project models the dynamics of a double pendulum using a Lagrangian neural network (LNN). The double pendulum is a classic example of a chaotic system in classical mechanics, where the motion is governed by nonlinear differential equations derived from the Lagrangian formulation of mechanics. The primary goal of this project is to generate training data from the mathematical model and use it to train an LNN to predict the future states of the system.

### Mathematical Model
#### Lagrangian Mechanics
The Lagrangian ( \( L \) ) is defined as the difference between the kinetic energy ( \( T \) ) and the potential energy ( \( V \) ) of the system:

<img src="https://latex.codecogs.com/svg.latex?L=T-V" title="L = T - V" />

#### Kinetic Energy ( \( T \) )
The kinetic energy of each pendulum is computed as follows:

For the first pendulum:

<img src="https://latex.codecogs.com/svg.latex?T_1=\frac{1}{2}m_1\left(l_1\dot{q}_1\right)^2" title="T_1 = \frac{1}{2} m_1 \left( l_1 \dot{q}_1 \right)^2" />

For the second pendulum:

<img src="https://latex.codecogs.com/svg.latex?T_2=\frac{1}{2}m_2\left(\left(l_1\dot{q}_1\cos(q_1)+l_2\dot{q}_2\cos(q_2)\right)^2+\left(l_1\dot{q}_1\sin(q_1)+l_2\dot{q}_2\sin(q_2)\right)^2\right)" title="T_2 = \frac{1}{2} m_2 \left( \left( l_1 \dot{q}_1 \cos(q_1) + l_2 \dot{q}_2 \cos(q_2) \right)^2 + \left( l_1 \dot{q}_1 \sin(q_1) + l_2 \dot{q}_2 \sin(q_2) \right)^2 \right)" />

#### Potential Energy ( \( V \) )
The potential energy of the pendulum system is given by:

For the first pendulum:

\[ V_1 = m_1 g l_1 (1 - \cos(q_1)) \]

For the second pendulum:

\[ V_2 = m_2 g \left( l_1 (1 - \cos(q_1)) + l_2 (1 - \cos(q_2)) \right) \]



### Equations of Motion

The equations of motion for the double pendulum can be derived from the Lagrangian using the Euler-Lagrange equations. These equations describe how the angles and angular velocities change over time, considering gravitational forces and the interaction between the two pendulums.

## Project Steps

1. **Set Up the Environment**:
   - Ensure you have Python installed along with the necessary libraries:
     ```bash
     pip install numpy torch pandas matplotlib
     ```

2. **Create the Data Directory**:
   - Create a directory named `data` in your project folder to store generated data.

3. **Generate Data**:
   - Use the provided `generate_data` function to simulate the double pendulum's motion over a specified time span and time step. The data will be saved as CSV files in the `data` folder.

4. **Train the Lagrangian Neural Network**:
   - Import the generated data from the CSV files in your LNN implementation file.
   - Define the LNN architecture and training loop to learn the dynamics of the double pendulum based on the input state and the true output state.

5. **Visualize Results**:
   - Implement visualization functions to display the motion of the double pendulum and plot the loss during training.

## Usage

To run the project, execute the main Python script, which will generate data, train the LNN, and visualize the results. Ensure that the necessary files and directories are in place as described in the steps above.

```bash
python main.py
```

## Conclusion

This project demonstrates the application of neural networks in learning the dynamics of nonlinear systems using Lagrangian mechanics. The trained LNN can be further utilized to predict future states of the double pendulum or to control similar systems.

---
