class ContinueModel(nn.Module):
  # def __init__(self, inputsize, activation, hidden_size, hidden_dim):
    # super().__init__()
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # self.network = build_nn(inputsize,1, activation, [hidden_size]*hidden_dim)
  def __init__(self, latent_length, latent_classes, deterministic_size, activation, hidden_size, hidden_dim):
    super().__init__()
    self.latent_size = latent_length*latent_classes
    self.input_size = deterministic_size+self.latent_size
    self.activation = activation
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.network = build_nn(self.inputsize, self.hidden_size,self.layer_size ,1, activation)


  def forward(self, x):
    return torch.distributions.Bernoulli(logits=self.network(x))
