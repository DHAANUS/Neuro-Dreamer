import torch.nn as nn
import torch

class BGCA(nn.Module):
  def __init__(self):
    super().__init__()
    self.k_proj = nn.Linear(128, 256)
    self.q_proj = nn.Linear(512, 256)
    self.v_proj = nn.Linear(128, 256)
    self.gate = nn.Sequential(nn.Linear(512, 1),
                              nn.Sigmoid())
    self.out = nn.Linear(256, 1024)

  def forward(self, prior, spatial_map, enc_out):

    q = self.q_proj(prior)
    q = q.unsqueeze(1)
    dist = spatial_map.view(-1, 16, 128)
    k = self.k_proj(dist)
    k = k.transpose(1, 2)
    v = self.v_proj(dist)

    dot_prod = q @ k
    summation = torch.nn.functional.softmax(dot_prod, -1)

    att = summation @ v
    sqd_att = att.squeeze(1)

    logit = self.out(sqd_att)

    gate = self.gate(prior)
    return enc_out + gate * logit
