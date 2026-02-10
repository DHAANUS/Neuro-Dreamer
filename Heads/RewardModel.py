class RewardModel(nn.Module):
  def __init__(self, latent_length, latent_classes, deterministic_size, activation, hidden_size, hidden_dim):
    super().__init__()
    self.latent_size = latent_length*latent_classes
    self.input_size = deterministic_size+self.latent_size
    self.activation = activation
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = build_nn(self.inputsize, self.hidden_size,self.layer_size ,2, activation)

  def forward(self, x):
    out = self.network(x)
    mean , logsd = out.chunk(2, dim=-1)
    sd = torch.exp(logsd)
    return torch.distributions.Normal(mean, sd)
