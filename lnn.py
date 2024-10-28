import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
from torch.optim import Adam
import pandas as pd
import os


class LagrangianNN(nn.Module):
    def __init__(self):
        super(LagrangianNN, self).__init__()
        self.fc1 = nn.Linear(4, 32)
        self.fc2 = nn.Linear(32, 32)
        self.fc3 = nn.Linear(32, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


# Load the input data and true output from CSV files
data_dir = './data'
input_data = pd.read_csv(os.path.join(data_dir, 'input_data.csv')).values
true_output = pd.read_csv(os.path.join(data_dir, 'true_output.csv')).values

# Convert to PyTorch tensors
input_tensor = torch.tensor(input_data, dtype=torch.float32)
true_output_tensor = torch.tensor(true_output, dtype=torch.float32)
# Loss function
def lagrangian_loss(model, input_data):
    predicted_output = model(input_data)
    loss = nn.MSELoss()(predicted_output, true_output)
    return loss

# Train the model
def train_model():
    model = LagrangianNN()
    optimizer = Adam(model.parameters(), lr=0.001)
    epochs = 1000
    loss_history = []

    for epoch in range(epochs):
        optimizer.zero_grad()
        loss = lagrangian_loss(model, input_data)
        loss.backward()
        optimizer.step()
        loss_history.append(loss.item())
        if (epoch + 1) % 100 == 0:
            print(f"Epoch [{epoch + 1}/{epochs}], Loss: {loss.item():.4f}")
    # Saving the weights
    torch.save(model.state_dict(), 'lnn_model_weights.pth')
    print("Model weights saved successfully.")
    return model, loss_history

# Train the model and visualize results
model, loss_history = train_model()

# Visualize the loss
plt.figure(figsize=(10, 5))
plt.plot(loss_history)
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid()
plt.show()

# Visualize the predictions
with torch.no_grad():
    predictions = model(input_data).numpy()

plt.figure(figsize=(10, 5))
plt.plot(true_output.numpy()[:, 0], label='True Theta1')
plt.plot(predictions[:, 0], label='Predicted Theta1', linestyle='dashed')
plt.plot(true_output.numpy()[:, 1], label='True Theta2')
plt.plot(predictions[:, 1], label='Predicted Theta2', linestyle='dashed')
plt.title('True vs Predicted States of Double Pendulum')
plt.xlabel('Time Step')
plt.ylabel('Angle (radians)')
plt.legend()
plt.grid()
plt.show()