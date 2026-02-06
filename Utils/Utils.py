import torch
import torch.nn as nn

def build_nn(input_size, hidden_size, num_layers ,output_size ,activation="Tanh"):
  assert num_layers>=2,"layers must be 2 or more"
  layers = []
  layers.append(nn.Linear(input_size, hidden_size))
  layers.append(activation)
  for i in range(num_layers):
    layers.append(nn.Linear(hidden_size, hidden_size))
    layers.append(activation)
  layers.append(nn.Linear(hidden_size, output_size))
  network = nn.Sequential(*layers)
  return network
