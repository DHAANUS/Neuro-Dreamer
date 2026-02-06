class Encoder(nn.Module):
  def __init__(self, inputshape, outputsize, activation):
    super().__init__()
    self.device = torch.device("cuda" if torch.cuda.is_available() else 'cpu')
    channels, height, width = inputshape
    self.outputsize = outputsize
    self.network = nn.Sequential(
        nn.Conv2d(channels, 16, kernel_size=4, stride=4, padding=1),
        activation,
        nn.Conv2d(16,32, kernel_size=4, stride=2, padding=1),
        activation,
        nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
        activation,
        nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
        activation,
        nn.Flatten(),
        nn.Linear(128*(height//16)*(width//16), outputsize),
        activation
    )
  def forward(self, x):
    return self.network(x)
