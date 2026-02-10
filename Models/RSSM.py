
class Recurrent_model(nn.Module):
  def __init__(self, action_dim, latent_classes, latent_length, deterministic_size, activation):
    super().__init__()
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.action_dim = action_dim
    self.deterministic_size = deterministic_size
    self.stochastic_size = latent_length*latent_classes
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
  def __init__(self,hidden_size, layers_size, deterministic_size,latent_length, latent_classes, activation):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    self.latent_length = latent_length
    self.latent_class = latent_classes
    self.latent_size = latent_classes*latent_length
    self.activation = activation
    self.deterministic_size = deterministic_size

    self.layers_size = layers_size
    self.hidden_size = hidden_size
    self.network = build_nn(
        self.deterministic_size,
        self.hidden_size,
        self.layers_size,
        self.latent_size,
        activation = self.activation
    )

  def forward(self, x):
      x = self.network(x)
      probability = x.view(-1, self.latent_length, self.latent_class).softmax(-1)
      uniform = torch.ones_like(probability)/self.latent_class
      final_probability = (1-0.01)*probability + 0.01*uniform
      logits = torch.distributions.utils.probs_to_logits(final_probability)
      sample = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
      return sample.view(-1, self.latent_size), logits


class Posterior(nn.Module):
  def __init__(self,layers_size,hidden_size, deterministic_size,latent_length, latent_classes, activation, obs_size):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else "cpu")
    self.latent_length = latent_length
    self.latent_class = latent_classes
    self.latent_size = latent_classes*latent_length
    self.activation = activation
    self.deterministic_size = deterministic_size
    self.obs_size = obs_size


    self.layers_size = layers_size
    self.hidden_size = hidden_size
    self.network = build_nn(
        self.deterministic_size+self.obs_size,
        hidden_size,
        layers_size,
        self.latent_size,
        activation = self.activation
    )

  def forward(self, x):
      # x = torch.cat((deterministic, obs), 1)
      x = self.network(x)
      probability = x.view(-1, self.latent_length, self.latent_class).softmax(-1)
      uniform = torch.ones_like(probability)/self.latent_class
      final_probability = (1-0.01)*probability + 0.01*uniform
      logits = torch.distributions.utils.probs_to_logits(final_probability)
      sample = torch.distributions.Independent(torch.distributions.OneHotCategoricalStraightThrough(logits=logits), 1).rsample()
      return sample.view(-1, self.latent_size), logits
