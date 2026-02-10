def build_neuralnet(input_size, out_size, activation, hidden_size, layer_size):
  layers = []
  layers.append(nn.Linear(input_size, hidden_size))
  layers.append(activation)
  for i in layer_size:
    layers.append(nn.Linear(hidden_size, hidden_size))
    layers.append(activation())
  layers.append(nn.Linear(hidden_size, out_size))
  network = nn.Sequential(*layers)
  return network
class Recurrent_model(nn.Module):
  def __init__(self, action_dim, deterministic_size, stochastic_size, activation):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.action_dim = action_dim
    self.deterministic_size = deterministic_size
    self.stochastic_size = stochastic_size
    self.activation = activation

    self.linear = nn.Linear(
        self.action_dim+self.stochastic_size, 200
    )
    self.recurrent = nn.GRUCell(200, deterministic_size)

  def forward(self, stochastic, deterministic, action):
    x = torch.cat((stochastic, action), 1)
    x = self.activation(self.linear(x))
    x = self.recurrent(x, deterministic)
    return x

class Prior(nn.Module):
  def __init__(self, deterministic_size,latent_length, latent_class, activation):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    self.latent_length = latent_length
    self.latent_class = latent_class
    self.latent_size = latent_class*latent_length
    self.activation = activation
    self.deterministic_size

    self.network = build_neuralnet(
        self.deterministic_size,
        self.latent_size,
        hidden_size = 200,
        layers_size = 2,
        activation = self.activation
    )

  def forward(self, deterministic):
      x = self.network(deterministic)
      probability = x.view(-1, self.latent_length, self.latent_class).softmax(-1)
      uniform = torch.ones_like(probability)/self.latent_class
      final_probability = (1-0.01)*probability + 0.01*uniform
      logits = torch.distributions.utils.probs_to_logits(final_probability)
      sample = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
      return sample.view(-1, self.latent_size), logits


class Posterior(nn.Module):
  def __init__(self, deterministic_size,latent_length, latent_class, activation, obs_size):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    self.latent_length = latent_length
    self.latent_class = latent_class
    self.latent_size = latent_class*latent_length
    self.activation = activation
    self.deterministic_size = deterministic_size
    self.obs_size = obs_size

    self.network = build_neuralnet(
        self.deterministic_size+self.obs_size,
        self.latent_size,
        hidden_size = 200,
        layer_size = 2,
        activation = self.activation
    )

  def forward(self, deterministic, obs):
      x = torch.cat((deterministic, obs), 1)
      x = self.network(x)
      probability = x.view(-1, self.latent_length, self.latent_class).softmax(-1)
      uniform = torch.ones_like(probability)/self.latent_class
      final_probability = (1-0.01)*probability + 0.01*uniform
      logits = torch.distributions.utils.probs_to_logits(final_probability)
      sample = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
      return sample.view(-1, self.latent_size), logits
