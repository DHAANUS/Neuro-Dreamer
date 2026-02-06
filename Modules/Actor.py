class Actor(nn.Module):
  def __init__(self, inputsize, actionspace, activation):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    self.network = build_network(
        inputsize,
        actionspace,
        activation,
        16*16
    )
  def forward(self, x):
    return torch.distributions.OneHotCategorical(logits=self.network(x))
