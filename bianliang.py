import torch
import numpy as np

input_size = 6
hidden_size1 =2
hidden_size2 =8
hidden_size3=32
output_size = 1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
my_nn = torch.nn.Sequential(
    torch.nn.Linear(input_size, hidden_size1),
    # torch.nn.ReLU6(),
    # torch.nn.Linear(hidden_size1, hidden_size2),
    # torch.nn.LeakyReLU(),
    torch.nn.Linear(hidden_size1, output_size)
).float().to(device)  # Move model to GPU

simplest_model = torch.nn.Sequential(
    # torch.nn.ReLU(),
    torch.nn.Linear(input_size, output_size)
    ).float().to(device)