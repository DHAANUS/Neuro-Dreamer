
class Recurrent_model(nn.Module):
  def __init__(self, action_dim, stochasticSize, deterministic_size, config):
    super().__init__()
    # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self.action_dim = action_dim
    self.deterministic_size = deterministic_size
    self.stochasticSize = stochasticSize
    self.activation = config.dreamer.recurrentModel.activation

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
  def __init__(self, deterministic_size, latent_length, latent_classes, config):
    super().__init__()
    self.config = config
    self.latent_length = latent_length
    self.latent_class = latent_classes
    self.stochasticSize = latent_classes*latent_length
    self.activation = config.dreamer.prior.activation
    self.deterministic_size = deterministic_size

    self.layers_size = config.dreamer.prior.hiddenLayers
    self.hidden_size = config.dreamer.prior.hiddenSize
    self.network = build_nn(
        self.deterministic_size,
        self.hidden_size,
        self.layers_size,
        self.stochasticSize,
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
  def __init__(self, inputSize, latent_length, latent_classes, config):
    super().__init__()
    self.config = config
    self.latent_length = latent_length
    self.latent_class = latent_classes
    self.latent_size = latent_classes*latent_length
    self.inputSize = inputSize


    self.layers_size = config.dreamer.posterior.hiddenLayers
    self.hidden_size = config.dreamer.posterior.hiddenSize
    self.network = build_nn(
        self.inputSize,
        self.hidden_size,
        self.layers_size,
        self.latent_size,
        activation = self.config.dreamer.posterior.activation
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
