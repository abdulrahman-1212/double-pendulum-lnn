# Double Pendulum LNN Project

## Mathematical Model

The dynamics of a double pendulum can be described using the Lagrangian formulation. The Lagrangian \( L \) is defined as the difference between the kinetic energy \( T \) and the potential energy \( V \) of the system:

\[
L = T - V
\]

### Kinetic Energy \( T \)

\[
T_1 = \frac{1}{2} m_1 \dot{q_1}^2 l_1^2
\]

\[
T_2 = \frac{1}{2} m_2 \left( \dot{q_1}^2 l_1^2 + \dot{q_2}^2 l_2^2 + 2 \dot{q_1} \dot{q_2} l_1 l_2 \cos(q_1 - q_2) \right)
\]

### Potential Energy \( V \)

\[
V_1 = m_1 g (l_1 (1 - \cos(q_1)))
\]

\[
V_2 = m_2 g (l_1 (1 - \cos(q_1)) + l_2 (1 - \cos(q_2)))
\]

### Project Steps

1. Set Up the Environment
2. Data Generation
3. Model Implementation
4. Training
5. Visualization
