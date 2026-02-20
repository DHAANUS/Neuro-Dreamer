import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import RIGHT_ONLY
from sys import exception
from re import template
import torch
import torch.nn as nn
import random
import numpy as np
import os
import yaml
import attridict
import csv
import pandas as pd
import plotly.graph_objects as pgo

def seeding(seed):
  random.seed(seed)
  np.random.seed(seed)
  torch.manual_seed(seed)
  torch.backends.cudnn.deterministic = True

def findFile(filename):
  currdirectory = os.getcwd()
  for root, dirs, files in os.walk(currdirectory):
    if filename in files:
      return os.path.join(root, filename)
  raise FileNotFoundError(f"{filename} not found in {currdirectory}")


def loadConfig(configPath):
  if not configPath.endswith(".yml"):
    configPath = configPath + ".yml"
  with open(configPath) as f:
    config = yaml.load(f, Loader=yaml.FullLoader)
  return attridict(config)

def saveLoss(filename, metrics):
  fileExist = os.path.isfile(filename + '.csv')
  with open(filename + '.csv', mode='a', newline = '') as file:
    writer = csv.writer(file)
    if not fileExist:
      writer.writerow(metrics.keys())
    writer.writerow(metrics.values())

def plotMetrics(filename, title='', savePath='metricsPlot', window=10):
  if not filename.endswith('.csv'):
    filename = filename + '.csv'
  data = pd.read_csv(filename)
  fig = pgo.Figure()

  colors = ["gold", "gray", "beige", "blueviolet", "cadetblue",
        "chartreuse", "coral", "cornflowerblue", "crimson", "darkorange",
        "deeppink", "dodgerblue", "forestgreen", "aquamarine", "lightseagreen",
        "lightskyblue", "mediumorchid", "mediumspringgreen", "orangered", "violet"]
  numColors = len(colors)
  for idx, column in enumerate(data.columns):
    if column in ['envSteps, gradientSteps']:
      continue
    fig.add_trace(pgo.Scatter(
        x=data['gradientSteps'],
        y=data[column],
        mode='lines',
        name=f'{column} (original)',
        line = dict(color='gray', width=1, dash='dot'),
        opacity = 0.5,
        visible='legendonly'
    ))
    smoothed = data[column].rolling(window=window, min_periods=1).mean()
    fig.add_trace(pgo.Scatter(
        x=data['gradientSteps'],
        y=smoothed,
        mode='lines',
        name=f'{column} (smoothed)',
        line = dict(color=colors[idx%numColors], width=2)
    ))
  fig.update_layout(
      title=dict(text=title,
                 x=0.5,
                 font=dict(size=30),
                 yanchor='top'),
      xaxis = dict(title='Gradient Step',
                   showgrid=True,
                   zeroline=False,
                   position=0),
      yaxis='Value',
      template = 'plotly_dark',
      height=1080,
      width=1920,
      margin=dict(t=60, l=40, r=40, b=40),
      legend = dict(x=0.02,
                    y=0.98,
                    xanchor='left',
                    yanchor='top',
                    bgcolor='rgba(0,0,0,0.5)',
                    bordercolor='white',
                    borderwidth=2,
                    font=dict(size=12)
                    )
  )
  if not savePath.endswith('.html'):
    savePath += '.html'
  fig.write_html(savePath)

def ensureParentFile(*paths):
  for path in paths:
    parentFolder = os.path.dirname(path)
    if parentFolder and not os.path.exists(parentFolder):
      os.makedirs(parentFolder, exist_ok=True)

def build_nn(input_size, hidden_size, num_layers ,output_size ,activation=False):
  layers = []
  layers.append(nn.Linear(input_size, hidden_size))
  layers.append(activation)
  for i in range(num_layers - 1):
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

def getEnvProperties(env):
  observationShape = env.observation_space.shape
  if isinstance(env.action_space, gym_super_mario_bros.actions.RIGHT_ONLY):
    discreteActionBool = True
    actionSize = env.action_space.n
  else:
    raise exception
  return observationShape, discreteActionBool, actionSize
class Moments(nn.Module):
  def __init__(self, device, decay = 0.99, min_=1, percentileLow = 0.05, percentileHigh=0.95):
    super().__init__()
    self.decay = decay
    self.min_ = torch.tensor(min_)
    self.percentileLow = percentileLow
    self.percentileHigh = percentileHigh
    self.register_buffer('low', torch.zeros((), dtype=torch.float32, device=device))
    self.register_buffer('high', torch.zeros((), dtype=torch.float32, device=device))

  def forward(self,x:torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    x = x.detach()
    low = torch.quantile(x, self.percentileLow)
    high = torch.quantile(x, self.percentileHigh)
    self.low = self.decay*self.low + (1-self.decay)*low
    self.high = self.decay*self.high + (1-self.decay)*high
    inverseScale = torch.max(self.min_, self.high - self.low)
    return self.low.detach(), inverseScale.detach()
