import torch
import torch.nn as nn

def build_nn(input_size, hidden_size, num_layers ,output_size ,activation=False):
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

def calc_lambda_values(rewards, values, continues, lambda_=0.95):
  returns = torch.zeros_like(rewards)
  bootstrap = values[:, -1]

  for i in reversed(range(rewards.shape[-1])):
    returns[:, i] = rewards[:, i] + continues[:, i] * ((1 - lambda_) * values[:, i] + lambda_ * bootstrap)
    bootstrap = returns[:, i]
  return returns
