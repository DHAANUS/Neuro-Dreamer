
class Decoder(nn.Module):
  def __init__(self, inputsize, outputshape, activation):
    super().__init__()
    self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    self.channels, self.height, self.width = outputshape
    self.network = nn.Sequential(
        nn.Linear(inputsize, 512),
        nn.Unflatten(1, (512, 1)),
        nn.Unflatten(2, (1, 1)),
        nn.ConvTranspose2d(512, 64, kernel_size=4, stride=2),
        activation,
        nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2),
        activation,
        nn.ConvTranspose2d(32, 16, kernel_size=4+1, stride=2),
        activation,
        nn.ConvTranspose2d(16, self.channels, kernel_size=4+1, stride=2),
    )
  def forward(self,x):
    return self.network(x)
